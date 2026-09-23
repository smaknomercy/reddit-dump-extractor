#!/usr/bin/env python3
import json
import os
import sys
import time
import re
from typing import Union, List, Set, Tuple, Optional

import pandas as pd
import psutil

from reddit_filter_utils import (
    parse_arguments, Config, MemoryMonitor, load_filter_values,
    collect_input_files, generate_output_path, FileReader, json_loads,
    DataNormalizer, setup_logging, build_prefilter, iter_candidate_lines,
    resolve_fields, format_eta
)


class OutputWriter:
    """
    Collects matched records and writes them out.

    Without `fields`: keeps every record in memory and writes once at the end
    (original behaviour, all columns).
    With `fields`: keeps only those columns and, for CSV, appends to the output
    file every `batch_size` records, so memory stays flat on huge files.
    """

    def __init__(self, output_path: str, output_format: str, config: Config,
                 fields: Optional[List[str]], batch_size: int):
        self.output_path = output_path
        self.output_format = output_format
        self.config = config
        self.fields = fields
        self.stream = fields is not None and output_format == 'csv'
        self.batch_size = batch_size
        self.buffer = []
        self.written = 0
        self.count = 0

    def add(self, obj: dict):
        if self.fields is not None:
            obj = {k: obj[k] for k in self.fields if k in obj}
        self.buffer.append(obj)
        self.count += 1
        if self.stream and len(self.buffer) >= self.batch_size:
            self._flush()

    def _frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.buffer, columns=self.fields) if self.fields else pd.DataFrame(self.buffer)
        return DataNormalizer.normalize_dataframe(df, self.config)

    def _flush(self):
        if not self.buffer:
            return
        df = self._frame()
        df.to_csv(
            self.output_path,
            mode='w' if self.written == 0 else 'a',
            header=self.written == 0,
            compression=self.config.get('output', 'csv_compression'),
            index=False
        )
        self.written += len(self.buffer)
        self.buffer = []

    def close(self):
        """Write whatever is left. Returns True if an output file exists."""
        if self.stream:
            self._flush()
            return self.written > 0
        if not self.buffer:
            return False
        df = self._frame()
        if self.output_format == 'parquet':
            df.to_parquet(
                self.output_path,
                engine='pyarrow',
                compression=self.config.get('output', 'parquet_compression'),
                index=False
            )
        else:
            df.to_csv(
                self.output_path,
                compression=self.config.get('output', 'csv_compression'),
                index=False
            )
        self.buffer = []
        return True


def process_file_python(
        file_path: str,
        field: str,
        values: Union[Set[str], List[re.Pattern]],
        regex: bool,
        output_path: str,
        output_format: str,
        config: Config,
        log,
        file_reader: FileReader,
        prefilter=None,
        fields: Optional[List[str]] = None,
        batch_size: int = 100000,
) -> Tuple[str, int, int, int]:
    """
    Process one .zst file using Python's zstandard library.

    With a prefilter, only lines whose raw bytes contain  "field":"value"
    are parsed; the exact check below is unchanged, so the matched records
    are the same. Malformed lines are then counted only among candidates.
    """
    writer = OutputWriter(output_path, output_format, config, fields, batch_size)
    lines_processed = 0
    error_lines = 0
    name = os.path.basename(file_path)

    value = None
    if len(values) == 1 and not regex:
        value = min(values)

    process = psutil.Process()
    process.cpu_percent()  # first call always returns 0.0; prime it
    log_interval = config.get('processing', 'progress_log_interval')
    next_log = log_interval
    start = time.time()

    def check(line: bytes):
        nonlocal error_lines
        try:
            obj = json_loads(line)
            observed = obj[field].lower()
            if regex:
                matched = any(reg.search(observed) for reg in values)
            elif value is not None:
                matched = observed == value
            else:
                matched = observed in values
            if matched:
                writer.add(obj)
        except (KeyError, json.JSONDecodeError, AttributeError):
            error_lines += 1

    try:
        for block, pos, total in file_reader.iter_blocks(file_path):
            if prefilter is not None:
                for line in iter_candidate_lines(block, prefilter):
                    check(line)
            else:
                for line in block.split(b'\n')[:-1]:
                    check(line)
            lines_processed += block.count(b'\n')

            if lines_processed >= next_log:
                next_log = (lines_processed // log_interval + 1) * log_interval
                elapsed = time.time() - start
                frac = pos / total if total else 0
                eta = format_eta(elapsed * (1 - frac) / frac) if frac > 0 else "?"
                mem = process.memory_info().rss / (1024 ** 3)
                log.info(
                    f"{name}: {frac:6.1%} | {lines_processed:,} lines, "
                    f"{writer.count:,} matched | ETA {eta} | "
                    f"CPU: {process.cpu_percent()}%, RAM: {mem:.2f} GB")

    except Exception as err:
        log.error(f"Error processing {file_path}: {err}")
        return file_path, lines_processed, 0, error_lines

    try:
        created = writer.close()
    except Exception as e:
        log.error(f"Failed to write output file {output_path}: {e}")
        return file_path, lines_processed, 0, error_lines

    if created:
        log.info(
            f"✓ Completed {name}: {lines_processed:,} lines, "
            f"{writer.count:,} matched, {error_lines:,} errors -> {output_path}")
    else:
        log.info(
            f"✓ Completed {name}: {lines_processed:,} lines, "
            f"0 matched, {error_lines:,} errors (no output file created)")

    return file_path, lines_processed, writer.count, error_lines


def main():
    args = parse_arguments()
    config = Config(args.config)
    log = setup_logging(config)

    fields = resolve_fields(args.fields, config)
    # .get with defaults: older config.json files don't have the new keys
    batch_size = args.batch_size or config._config.get('processing', {}).get('batch_size', 100000)

    log.info("=" * 80)
    log.info("Method 1 (Zstandard) - Reddit Dump Filter")
    log.info("=" * 80)
    log.info(f"Input directory: {args.input}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Output format: {args.format}")
    log.info(f"Field: {args.field} | Value: {args.value} | Regex: {args.regex}")
    if fields:
        mode = f"streaming, batch {batch_size:,}" if args.format == 'csv' else "in memory"
        log.info(f"Fields ({len(fields)}, {mode}): {', '.join(fields)}")
    log.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    memory_monitor = MemoryMonitor()
    file_reader = FileReader(config)
    values = load_filter_values(args, log)
    prefilter = None
    if args.no_prefilter:
        log.info("Prefilter disabled: --no_prefilter")
    else:
        prefilter = build_prefilter(args.field, values, args.regex, log)

    log.info(f"Scanning for input files matching pattern: {args.file_filter}")
    input_files = collect_input_files(args.input, args.file_filter, config)
    log.info(f"Found {len(input_files)} total files")

    if len(input_files) == 0:
        log.error("No matching files found!")
        sys.exit(1)

    total_processed = 0
    total_lines = 0
    total_matched = 0
    total_errors = 0
    start_time = time.time()

    log.info("=" * 80)
    log.info("Starting processing...")
    log.info("=" * 80)

    try:
        for input_file in input_files:
            output_path = generate_output_path(
                input_file, args.output_dir, args.format, config)
            file_path, lines_processed, matched_count, error_count = process_file_python(
                input_file,
                args.field,
                values,
                args.regex,
                output_path,
                args.format,
                config,
                log,
                file_reader,
                prefilter=prefilter,
                fields=fields,
                batch_size=batch_size,
            )

            total_processed += 1
            total_lines += lines_processed
            total_matched += matched_count
            total_errors += error_count

            progress_pct = (total_processed / len(input_files)) * 100
            mem_stats = memory_monitor.get_usage_stats()
            log.info(
                f"Progress: {total_processed}/{len(input_files)} ({progress_pct:.1f}%) | "
                f"Total matched: {total_matched:,} | RAM: {mem_stats['rss_gb']:.2f} GB"
            )

    except KeyboardInterrupt:
        log.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error during processing: {e}")
        raise

    elapsed = time.time() - start_time
    log.info("=" * 80)
    log.info("Processing Complete!")
    log.info("=" * 80)
    log.info(f"Files processed: {total_processed}")
    log.info(f"Total lines scanned: {total_lines:,}")
    log.info(f"Total records matched: {total_matched:,}")
    log.info(f"Total errors: {total_errors:,}" + (" (among prefiltered candidates)" if prefilter else ""))
    log.info(f"Elapsed time: {elapsed:.1f} seconds")
    if total_lines > 0:
        log.info(f"Processing rate: {total_lines / elapsed:.0f} lines/second")
    log.info(f"Output directory: {args.output_dir}")

    if args.format == 'csv':
        csv_compression = config.get('output', 'csv_compression')
        ext = '.csv.gz' if csv_compression == 'gzip' else '.csv'
    else:
        ext = '.parquet'
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith(ext)]
    log.info(f"Output files created: {len(output_files)}")

    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in output_files
    ) if output_files else 0
    log.info(f"Total output size: {total_size / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    main()
