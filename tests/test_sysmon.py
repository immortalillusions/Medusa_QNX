"""Pi usage: sampled by medusa_sysmon.py in its own process, read by the control HTTP thread."""

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import medusa_control
import medusa_sysmon
from medusa import sysmon
from medusa.control import ControlHTTPServer, ControlLoop, SystemStatusFile
from medusa.sysmon import (
    SYSTEM_KEYS, SystemSampler, cpu_from_times, find_pid, parse_cpu_seconds, parse_pidin_info, parse_pidin_times,
    parse_thermal, split_output, write_atomic,
)
from test_control import FakeClock, FakeSerial

# Captured shapes of the QNX 8 tools on the Pi.
PIDIN_INFO = ("CPU:AARCH64 Release:8.0.0  FreeMem:7069MB/8128MB BootTime:Nov 22 05:30:12 UTC 2026\n"
              "Processes: 61, Threads: 297\n"
              "Processor1: 1091555328 Cortex-A76 2400MHz FPU\n")
PIDIN_ARG = ("     pid Arguments\n"
             "       1 procnto-smp-instr -v\n"
             "  6275100 sh ./start_medusa.sh\n"
             "  6410270 on -p 21 python3 -u medusa_control.py --port /dev/ser1\n"
             "  6410282 python3 -u qnx_ai_service.py --control-port 8768\n"
             "  6410283 /system/bin/python3 -u /data/medusa/medusa_control.py --port /dev/ser1\n"
             "  6410290 /system/bin/python3 -c import sys, time\n")
TIMES_HEADER = "     pid name                       sid start time       utime     stime    cutime    cstime\n"


def pidin_times(procnto, ai, control, stress, io_sock, extra=""):
    return (TIMES_HEADER
            + f"       1 proc/boot/procnto-smp-instr  0 Nov 22 10:34 {procnto}   2m10s     0.000     0.000\n"
            + "  6275100 system/bin/sh        6275100 Nov 22 11:08     0.010     0.020     0.000     0.000\n"
            + f"  6410282 system/bin/python3   6275100 Nov 22 11:08 {ai}     1.500     0.000     0.000\n"
            + f"  6410283 system/bin/python3   6275100 Nov 22 11:08 {control}     2.000     0.000     0.000\n"
            + f"  6410290 system/bin/python3   6410283 Nov 22 11:41 {stress}     0.000     0.000     0.000\n"
            + f"     8200 proc/boot/io-sock          0 Nov 22 10:34 {io_sock}     3.000     0.000     0.000\n"
            + extra)


TIMES_A = pidin_times("14m40s", "68.071", "10.000", "1.000", "1m05s",
                      "  6410300 system/bin/ls        6275100 Nov 22 11:42     0.001     0.001     0.000     0.000\n")
TIMES_B = pidin_times("14m42s", "72.071", "10.400", "2.600", "1m06s",  # ls ended, cat started
                      "  6410301 system/bin/cat       6275100 Nov 22 11:42     0.500     0.000     0.000     0.000\n")


def output(times):
    """One spawn's output, as SYSTEM_COMMAND prints it."""
    return f"{PIDIN_INFO}@@medusa-arg\n{PIDIN_ARG}@@medusa-times\n{times}"


class ParserTests(unittest.TestCase):
    def test_cpu_time_forms(self) -> None:
        for token, seconds in (("68.071", 68.071), ("0.000", 0.0), ("14m40s", 880.0), ("2h05m", 7500.0),
                               ("1d03h", 97200.0), ("40s", 40.0), ("3m", 180.0)):
            with self.subTest(token=token):
                self.assertAlmostEqual(parse_cpu_seconds(token), seconds)
        for token in ("", "N/A", "11:08", "12x", "m40s"):
            self.assertIsNone(parse_cpu_seconds(token))

    def test_pidin_times_rows(self) -> None:
        times = parse_pidin_times(TIMES_A)
        self.assertEqual(times[1], ("proc/boot/procnto-smp-instr", 880.0 + 130.0))
        self.assertAlmostEqual(times[6410282][1], 69.571)
        self.assertAlmostEqual(times[8200][1], 68.0)
        self.assertEqual(len(times), 7)
        self.assertEqual(parse_pidin_times("6410282 python3 1 Nov 22 11:08 1.0 1.0 0 0"), {})  # no header
        live = parse_pidin_times(TIMES_HEADER + "6410282 system/bin/python3 6275100 Nov 22 11:08 68.071 0.000 0.000 0.000")
        self.assertEqual(live, {6410282: ("system/bin/python3", 68.071)})

    def test_pids_come_from_the_command_lines(self) -> None:
        self.assertEqual(find_pid(PIDIN_ARG, "medusa_control.py"), 6410283)  # not the `on` launcher
        self.assertEqual(find_pid(PIDIN_ARG, "qnx_ai_service.py"), 6410282)
        self.assertIsNone(find_pid(PIDIN_ARG, "laptop_detector.py"))
        self.assertIsNone(find_pid("sh -c pidin arg medusa_control.py", "medusa_control.py"))
        self.assertEqual(split_output(output(TIMES_A))[1], "\n" + PIDIN_ARG)

    def test_cpu_split_is_percent_of_the_whole_chip(self) -> None:
        before, after = parse_pidin_times(TIMES_A), parse_pidin_times(TIMES_B)
        # 2 s on 4 cores = 8 core-seconds. AI +4.0 s, control +0.4, stress +1.6, io-sock +1.0; procnto
        # skipped; ls ended and cat started between the samples: both skipped.
        self.assertEqual(cpu_from_times(before, after, 2.0, 4, 6410283, 6410282), (87.5, 5.0, 50.0))
        self.assertEqual(cpu_from_times(before, after, 2.0, 4, None, None), (87.5, None, None))
        self.assertEqual(cpu_from_times(before, after, 0.01, 4, 6410283, 6410282)[0], 100.0)
        self.assertEqual(cpu_from_times({}, after, 2.0, 4, 6410283, 6410282), (None, None, None))
        reused = {**after, 6410282: ("system/bin/sh", 1.0)}  # a reused pid is skipped, not negative
        self.assertIsNone(cpu_from_times(before, reused, 2.0, 4, 6410283, 6410282)[2])

    def test_pidin_info_and_thermal(self) -> None:
        free, total, boot = parse_pidin_info(PIDIN_INFO)
        self.assertEqual((free, total), (7069.0, 8128.0))
        self.assertEqual(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(boot)), "2026-11-22 05:30:12")
        self.assertEqual(parse_pidin_info("FreeMem:7G/8G")[:2], (7168.0, 8192.0))
        self.assertEqual(parse_pidin_info("nothing here"), (None, None, None))
        self.assertEqual(parse_thermal("51.57\n"), 51.6)
        self.assertIsNone(parse_thermal(""))
        self.assertIsNone(parse_thermal("9999"))


class SamplerTests(unittest.TestCase):
    def test_sample_combines_sources_cpu_needs_two_samples(self) -> None:
        clock = FakeClock(50.0)
        boot = parse_pidin_info(PIDIN_INFO)[2]
        outputs = [output(TIMES_A), output(TIMES_B)]
        sampler = SystemSampler(run=lambda: outputs.pop(0), read_thermal=lambda: "51.57", cores=4,
                                clock=clock.monotonic, wall=lambda: boot + 3600)
        first = sampler.sample_once()
        self.assertEqual((first["cpu_total_pct"], first["cpu_control_pct"], first["cpu_ai_pct"]), (None,) * 3)
        clock.now = 52.0
        self.assertEqual(sampler.sample_once(), {"cpu_total_pct": 87.5, "cpu_control_pct": 5.0, "cpu_ai_pct": 50.0,
                                                 "mem_free_mb": 7069.0, "mem_total_mb": 8128.0, "temp_c": 51.6,
                                                 "uptime_s": 3600})

    def test_cpu_base_is_the_oldest_sample_inside_the_window(self) -> None:
        clock = FakeClock(0.0)
        texts = [output(t) for t in (TIMES_A, TIMES_A, TIMES_B, TIMES_B)]
        sampler = SystemSampler(run=lambda: texts.pop(0), read_thermal=lambda: "", cores=4, window_s=10.0,
                                clock=clock.monotonic)
        for now in (0.0, 2.0, 4.0):
            clock.now = now
            sample = sampler.sample_once()
        self.assertEqual(sample["cpu_ai_pct"], 25.0)  # 4.0 s over 4 s x 4 cores, against t=0
        clock.now = 13.5  # t=0 and t=2 fell out of the window; t=4 is the base now
        self.assertEqual(sampler.sample_once()["cpu_ai_pct"], 0.0)

    def test_failures_are_null_never_raised(self) -> None:
        def boom():
            raise OSError("no such tool")
        with patch.object(sysmon, "on_qnx", return_value=False):
            sample = SystemSampler(run=boom, read_thermal=boom).sample_once()
        self.assertEqual(sample, dict.fromkeys(k for k in SYSTEM_KEYS if k != "sampled_age_s"))

    def test_off_qnx_nothing_is_spawned_and_on_qnx_one_spawn_at_priority_10(self) -> None:
        with patch.object(sysmon, "on_qnx", return_value=False), patch.object(sysmon.subprocess, "run") as run:
            self.assertEqual(sysmon._run_system_command(), "")
        run.assert_not_called()
        with patch.object(sysmon, "on_qnx", return_value=True), \
                patch.object(sysmon.shutil, "which", return_value="/proc/boot/on"), \
                patch.object(sysmon.subprocess, "run", return_value=SimpleNamespace(stdout=b"x")) as run:
            self.assertEqual(sysmon._run_system_command(), "x")
        self.assertEqual(run.call_args.args[0][:3], ["on", "-p", "10"])
        self.assertEqual(run.call_count, 1)


class FileRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "medusa_system.json")
        self.sample = {"cpu_total_pct": 40.1, "cpu_control_pct": 2.4, "cpu_ai_pct": 30.0, "mem_free_mb": 7069.0,
                       "mem_total_mb": 8128.0, "temp_c": 51.6, "uptime_s": 3600}

    def test_written_sample_is_served_with_its_age(self) -> None:
        write_atomic(self.path, self.sample)
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["medusa_system.json"])  # no temp left
        mtime = os.stat(self.path).st_mtime
        reader = SystemStatusFile(self.path, wall=lambda: mtime + 1.5)
        self.assertEqual(reader.read(), {**self.sample, "sampled_age_s": 1.5})

    def test_missing_stale_or_garbage_is_all_null(self) -> None:
        empty = dict.fromkeys(SYSTEM_KEYS)
        self.assertEqual(SystemStatusFile(self.path).read(), empty)  # missing
        write_atomic(self.path, self.sample)
        mtime = os.stat(self.path).st_mtime
        self.assertEqual(SystemStatusFile(self.path, wall=lambda: mtime + 10.5).read(), empty)  # stale
        self.assertEqual(SystemStatusFile(self.path, wall=lambda: mtime + 9.5).read()["cpu_ai_pct"], 30.0)
        Path(self.path).write_text("{half")
        self.assertEqual(SystemStatusFile(self.path).read(), empty)
        Path(self.path).write_text('{"cpu_ai_pct": "lots", "temp_c": 50}')
        self.assertEqual(SystemStatusFile(self.path).read()["cpu_ai_pct"], None)  # wrong types are null

    def test_reader_caches_by_mtime(self) -> None:
        write_atomic(self.path, self.sample)
        reader = SystemStatusFile(self.path)
        reader.read()
        with patch("builtins.open", side_effect=AssertionError("re-read an unchanged file")):
            self.assertEqual(reader.read()["cpu_ai_pct"], 30.0)
        write_atomic(self.path, {**self.sample, "cpu_ai_pct": 31.0})
        os.utime(self.path, ns=(time.time_ns(), time.time_ns() + 1000))  # a new mtime even within one tick
        self.assertEqual(reader.read()["cpu_ai_pct"], 31.0)

    def test_sysmon_loop_writes_the_file(self) -> None:
        sampler = SystemSampler(run=lambda: output(TIMES_A), read_thermal=lambda: "40", period_s=0.01)
        sampler.run_forever(self.path, threading.Event(), max_samples=2)
        self.assertEqual(json.loads(Path(self.path).read_text())["temp_c"], 40.0)

    def test_sysmon_main_parses_flags(self) -> None:
        with patch.object(medusa_sysmon, "SystemSampler") as made, patch("builtins.print"), \
                patch.object(medusa_sysmon.signal, "signal"):
            self.assertEqual(medusa_sysmon.main(["--out", self.path, "--period", "3"]), 0)
        self.assertEqual(made.call_args.kwargs["period_s"], 3.0)
        self.assertEqual(made.return_value.run_forever.call_args.args[0], self.path)


class ControlServesTheFileTests(unittest.TestCase):
    def test_control_json_system_block_comes_from_the_file_and_the_loop_never_reads_it(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "medusa_system.json")
        write_atomic(path, {"cpu_total_pct": 12.5, "temp_c": 48.0})
        loop = ControlLoop(FakeSerial())
        with patch.object(SystemStatusFile, "read", side_effect=AssertionError("loop read the file")):
            for k in range(20):
                loop.tick(k * 0.02)
            self.assertNotIn("system", loop.snapshot())  # the snapshot (and so the loop) never has it
        self.assertFalse(hasattr(loop, "system"))
        server = ControlHTTPServer(loop, port=0, host="127.0.0.1", system_file=SystemStatusFile(path))
        server.start()
        self.addCleanup(server.stop)
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/control.json", timeout=5) as response:
            system = json.loads(response.read())["system"]
        self.assertEqual(set(system), set(SYSTEM_KEYS))
        self.assertEqual((system["cpu_total_pct"], system["temp_c"], system["cpu_ai_pct"]), (12.5, 48.0, None))

    def test_control_process_has_no_sampler_and_takes_the_file_flag(self) -> None:
        self.assertEqual(medusa_control.parse_args([]).system_file, "/tmp/medusa_system.json")
        self.assertEqual(medusa_control.parse_args(["--system-file", "/x.json"]).system_file, "/x.json")
        self.assertNotIn("SystemSampler", vars(medusa_control))


if __name__ == "__main__":
    unittest.main()
