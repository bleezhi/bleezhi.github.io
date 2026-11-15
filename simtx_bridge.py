#!/usr/bin/env python3
"""
SimTX - rtl_tcp -> YouTube Live bridge.

* Real FFT spectrum + waterfall rendered from the wideband IQ stream
* "SimTX" watermark top-right, live frequency/mode readout top-left
* Runtime control over a UNIX socket (retune without restarting the stream)
* Presets + optional auto-rotation, rtl_tcp auto-reconnect (stream stays up)
* Click-free streaming demodulators (filter state is kept between chunks)

Control examples (while the service is running):
    simtx_bridge.py --send "freq 145.5M"
    simtx_bridge.py --send "freq 7.1M USB; gain 30"
    simtx_bridge.py --send "next"            # next preset
    simtx_bridge.py --send "rotate 60"       # cycle presets every 60 s ("rotate off")
    simtx_bridge.py --send "status"
"""
import argparse
import os
import queue
import re
import shutil
import signal as pysignal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from scipy import signal as dsp

# --- DEFAULT CONFIGURATION ---
RTL_HOST = "argon.simtx.net"
RTL_PORT = 1240
SAMPLE_RATE = 1_024_000          # 1.024 MSPS
AUDIO_RATE = 32_000              # PCM rate handed to ffmpeg
CHUNKS_PER_SEC = 4
CHUNK_SAMPLES = SAMPLE_RATE // CHUNKS_PER_SEC      # 256000 IQ samples = 250 ms
AUDIO_PER_CHUNK = AUDIO_RATE // CHUNKS_PER_SEC     # 8000
MODES = ["NFM", "WFM", "AM", "USB", "LSB", "RAW"]
BRAND = "SimTX"

FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
_FREQ_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*([gmk]?)(?:hz)?$")


def parse_freq(val: str) -> int:
    """'144.8M', '7150k', '1.2g', '144800000' -> Hz. A bare number below 100000 is taken as MHz."""
    m = _FREQ_RE.match(val.strip().lower())
    if not m:
        raise ValueError(f"bad frequency: {val!r}")
    num, unit = float(m.group(1)), m.group(2)
    if not unit and num < 100_000:
        unit = "m"
    return int(round(num * {"": 1, "k": 1e3, "m": 1e6, "g": 1e9}[unit]))


def parse_mode(val: str) -> str:
    v = val.strip().upper()
    if v not in MODES:
        raise ValueError(f"bad mode {val!r} (choose from {', '.join(MODES)})")
    return v


def parse_presets(text: str, default_mode: str):
    """'144.8M:NFM,145.5M,98.5M:WFM' -> [(Hz, mode), ...]"""
    out = []
    for item in filter(None, (p.strip() for p in (text or "").split(","))):
        f, _, m = item.partition(":")
        out.append((parse_freq(f), parse_mode(m) if m else default_mode))
    return out


def default_sock_path() -> str:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~")
    return os.path.join(base, "simtx.sock")


def find_font(user_font):
    if user_font:
        return user_font
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bridge a remote rtl_tcp stream to YouTube Live with FFT waterfall, "
        "software AGC, runtime retuning and a SimTX watermark."
    )
    p.add_argument("-f", "--freq", type=parse_freq, default="144.8M",
                   help="Start frequency (144.8M, 7150k, 98.5, or Hz). Default 144.8M")
    p.add_argument("-m", "--mode", type=str.upper, choices=MODES, default="NFM",
                   help="Start demodulation mode. Default NFM")
    p.add_argument("-g", "--gain", type=float, default=40.0,
                   help="Tuner gain in dB (0 = auto). Default 40")
    p.add_argument("--agc", dest="agc", action="store_true", default=True,
                   help="Enable software digital AGC (default)")
    p.add_argument("--no-agc", dest="agc", action="store_false",
                   help="Disable software digital AGC")
    p.add_argument("--ppm", type=int, default=0, help="Frequency correction in ppm")
    p.add_argument("--direct-sampling", choices=["off", "i", "q", "auto"], default="off",
                   help="rtl_tcp direct sampling for HF ('auto' = Q-branch below 24 MHz)")
    p.add_argument("--host", default=RTL_HOST, help=f"rtl_tcp host. Default {RTL_HOST}")
    p.add_argument("--port", type=int, default=RTL_PORT, help=f"rtl_tcp port. Default {RTL_PORT}")

    p.add_argument("--presets", default="",
                   help="Comma list 'freq[:mode],...' e.g. '144.8M:NFM,145.5M,98.5M:WFM'")
    p.add_argument("--rotate", type=float, default=0.0,
                   help="Seconds between automatic preset changes (0 = off)")
    p.add_argument("--ctl-sock", default=default_sock_path(),
                   help="UNIX control socket path. Default %(default)s")
    p.add_argument("--send", metavar="CMD",
                   help="Send command(s) to a running instance (';' separated) and exit")

    p.add_argument("--size", default="1280x720", help="Video size. Default 1280x720")
    p.add_argument("--fps", type=int, default=8, help="Video fps, multiple of 4. Default 8")
    p.add_argument("--span", type=parse_freq, default="160k",
                   help="Spectrum span around centre frequency. Default 160k")
    p.add_argument("--db-range", type=float, default=45.0, help="Display dynamic range in dB. Default 45")
    p.add_argument("--bitrate", type=int, default=1500, help="Video bitrate in kbps. Default 1500")
    p.add_argument("--x264-preset", default="veryfast", help="x264 preset. Default veryfast")
    p.add_argument("--font", default=None, help="TTF font file for the text overlay (auto-detected)")
    p.add_argument("--rtmp-url", default="rtmp://a.rtmp.youtube.com/live2",
                   help="RTMP base URL (stream key comes from YT_STREAM_KEY)")
    args = p.parse_args()

    try:
        args.w, args.h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        p.error("--size must look like 1280x720")
    if args.fps % CHUNKS_PER_SEC:
        p.error("--fps must be a multiple of 4")
    if not 20_000 <= args.span <= int(SAMPLE_RATE * 0.9):
        p.error("--span must be between 20k and ~900k")
    return args


# ----------------------------------------------------------------------------
# DSP
# ----------------------------------------------------------------------------
class DigitalAGC:
    """Software Digital AGC for dynamic audio volume normalization."""

    def __init__(self, target_level=0.5, alpha=0.002, max_gain=40.0):
        self.target_level = target_level
        self.alpha = alpha
        self.max_gain = max_gain
        self.current_gain = 1.0

    def process(self, audio_block):
        mean_env = max(float(np.mean(np.abs(audio_block))), 1e-6)
        error = self.target_level - mean_env
        self.current_gain += self.alpha * error
        self.current_gain = float(np.clip(self.current_gain, 0.1, self.max_gain))
        return audio_block * self.current_gain


class StreamDecimator:
    """FIR decimator that keeps history between chunks (no edge clicks, polyphase-cheap)."""

    def __init__(self, q: int, cutoff: float = 0.9):
        self.q = q
        self.h = dsp.firwin(16 * q + 1, cutoff / q).astype(np.float32)
        self.L = len(self.h) - 1          # multiple of q by construction
        self.hist = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.hist is None:
            self.hist = np.zeros(self.L, dtype=x.dtype)
        buf = np.concatenate((self.hist, x))
        self.hist = buf[-self.L:]
        y = dsp.upfirdn(self.h, buf, up=1, down=self.q)
        s = self.L // self.q
        return y[s:s + len(x) // self.q].astype(x.dtype)


class Demod:
    SCALE = {"RAW": 100.0, "WFM": 8000.0, "AM": 600.0, "USB": 3000.0, "LSB": 3000.0, "NFM": 12000.0}

    def __init__(self, mode: str):
        self.mode = mode
        self.reset()

    def reset(self):
        self.dec1 = StreamDecimator(8)     # 1.024M -> 128k
        self.dec2 = StreamDecimator(4)     # 128k  -> 32k
        self.prev = np.complex64(0)
        self.am_zi = None
        self.ssb_h = self.ssb_zi = None
        if self.mode in ("USB", "LSB"):
            # complex band-pass 300..2700 Hz: low-pass shifted to +/-1500 Hz
            n = np.arange(255)
            lp = dsp.firwin(255, 1200.0, fs=AUDIO_RATE)
            sign = 1.0 if self.mode == "USB" else -1.0
            self.ssb_h = (lp * np.exp(sign * 2j * np.pi * 1500.0 * n / AUDIO_RATE)).astype(np.complex64)
            self.ssb_zi = np.zeros(254, dtype=np.complex64)

    def _fm(self, x):
        ext = np.concatenate(([self.prev], x))
        self.prev = x[-1]
        return np.angle(ext[1:] * np.conj(ext[:-1])).astype(np.float32)

    def _am(self, env):
        b, a = [1.0, -1.0], [1.0, -0.995]          # DC blocker
        if self.am_zi is None:
            self.am_zi = dsp.lfilter_zi(b, a) * env[0]
        y, self.am_zi = dsp.lfilter(b, a, env, zi=self.am_zi)
        return y

    def process(self, iq: np.ndarray) -> np.ndarray:
        m = self.mode
        if m == "RAW":
            return np.real(iq[::32]).astype(np.float32) * self.SCALE[m]
        x = self.dec1.process(iq)
        if m == "WFM":
            audio = self.dec2.process(self._fm(x))
        else:
            x = self.dec2.process(x)
            if m == "NFM":
                audio = self._fm(x)
            elif m == "AM":
                audio = self._am(np.abs(x))
            else:  # USB / LSB
                y, self.ssb_zi = dsp.lfilter(self.ssb_h, [1.0], x, zi=self.ssb_zi)
                audio = y.real
        return audio.astype(np.float32) * self.SCALE[m]


# ----------------------------------------------------------------------------
# FFT + waterfall rendering
# ----------------------------------------------------------------------------
class Spectrum:
    """Averaged FFT of an IQ segment, cropped to +/- span/2 and resampled to `width` columns."""

    def __init__(self, width: int, span: int, n_fft: int = 8192):
        self.n = n_fft
        self.width = width
        self.window = np.hanning(n_fft).astype(np.float32)
        half = max(8, int(span / SAMPLE_RATE * n_fft / 2))
        self.lo, self.hi = n_fft // 2 - half, n_fft // 2 + half

    def row(self, iq: np.ndarray) -> np.ndarray:
        nb = max(1, len(iq) // self.n)
        x = iq[: nb * self.n].reshape(nb, self.n) * self.window
        p = np.mean(np.abs(np.fft.fft(x, axis=1)) ** 2, axis=0)
        p = np.fft.fftshift(p)
        c = self.n // 2
        p[c - 1:c + 2] = 0.5 * (p[c - 3] + p[c + 3])       # kill the RTL DC spike
        db = 10.0 * np.log10(p + 1e-9)[self.lo:self.hi]
        if len(db) == self.width:
            return db.astype(np.float32)
        return np.interp(np.linspace(0, len(db) - 1, self.width), np.arange(len(db)), db).astype(np.float32)


def _make_lut() -> np.ndarray:
    pts = [0.0, 0.2, 0.45, 0.7, 0.9, 1.0]
    r = [0, 0, 0, 250, 255, 255]
    g = [0, 0, 190, 230, 70, 255]
    b = [16, 170, 220, 40, 0, 255]
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, pts, ch) for ch in (r, g, b)], axis=1).astype(np.uint8)


LUT = _make_lut()


class Renderer:
    HDR = 64          # header strip reserved for the text overlay
    ROW_H = 2         # waterfall pixels per FFT row

    def __init__(self, w: int, h: int, db_range: float):
        self.w, self.h, self.range = w, h, db_range
        self.spec_h = int(h * 0.30)
        self.wf_h = h - self.HDR - self.spec_h
        self.wf = np.zeros((self.wf_h, w, 3), np.uint8)
        self.floor = None
        self.cols = np.arange(w)
        self.rows = np.arange(self.spec_h)[:, None]
        bg = np.empty((self.spec_h, w, 3), np.uint8)
        bg[:] = (12, 12, 20)
        for k in range(1, 8):
            bg[:, k * w // 8] = (36, 36, 52)
        for k in range(1, 4):
            bg[k * self.spec_h // 4, :] = (30, 30, 44)
        self.bg = bg

    def push(self, db=None) -> bytes:
        if db is None:                       # no signal: black row, flat trace
            lv = np.zeros(self.w, np.float32)
        else:
            p20 = float(np.percentile(db, 20))
            self.floor = p20 if self.floor is None else 0.9 * self.floor + 0.1 * p20
            lo = self.floor - 4.0
            lv = np.clip((db - lo) / self.range, 0.0, 1.0)

        row = LUT[(lv * 255).astype(np.uint8)]
        self.wf = np.concatenate((np.repeat(row[None], self.ROW_H, axis=0), self.wf[:-self.ROW_H]))

        spec = self.bg.copy()
        top = self.spec_h - 1 - (lv * (self.spec_h - 3)).astype(np.int32)
        spec[self.rows >= top[None, :]] = (25, 70, 140)
        spec[top, self.cols] = (120, 230, 255)

        frame = np.empty((self.h, self.w, 3), np.uint8)
        frame[: self.HDR] = (18, 18, 26)
        frame[self.HDR:self.HDR + self.spec_h] = spec
        frame[self.HDR + self.spec_h:] = self.wf
        c = self.w // 2                       # centre-frequency marker
        col = frame[self.HDR:, c].astype(np.uint16)
        frame[self.HDR:, c] = ((col * 2 + np.array([255, 0, 0], np.uint16)) // 3).astype(np.uint8)
        return frame.tobytes()


# ----------------------------------------------------------------------------
# rtl_tcp
# ----------------------------------------------------------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("rtl_tcp server closed the connection.")
        buf.extend(chunk)
    return bytes(buf)


class RtlSource:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.sock = None

    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=5)
        s.settimeout(10)
        try:
            if not recv_exact(s, 12).startswith(b"RTL0"):
                raise ConnectionError("unexpected rtl_tcp header")
        except Exception:
            s.close()
            raise
        self.sock = s
        self.send(0x02, SAMPLE_RATE)

    def send(self, cmd: int, param: int):
        self.sock.sendall(struct.pack(">BI", cmd, param & 0xFFFFFFFF))

    def read(self) -> bytes:
        return recv_exact(self.sock, CHUNK_SAMPLES * 2)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


# ----------------------------------------------------------------------------
# runtime control
# ----------------------------------------------------------------------------
def parse_command(text: str, bridge) -> str:
    """Parse 'cmd args; cmd args' and queue the changes for the main loop."""
    q, replies = bridge.cmd_q, []
    for part in text.split(";"):
        t = part.split()
        if not t:
            continue
        c = t[0].lower()
        try:
            if c == "status":
                replies.append(bridge.status)
                continue
            if c == "help":
                replies.append("freq <f> [mode] | mode <m> | gain <dB> | agc on|off | next | prev | "
                               "preset <n> | rotate <sec|off> | status")
                continue
            if c == "freq" and len(t) >= 2:
                q.put(("freq", parse_freq(t[1])))
                if len(t) >= 3:
                    q.put(("mode", parse_mode(t[2])))
            elif c == "mode" and len(t) >= 2:
                q.put(("mode", parse_mode(t[1])))
            elif c == "gain" and len(t) >= 2:
                q.put(("gain", float(t[1])))
            elif c == "agc" and len(t) >= 2:
                q.put(("agc", t[1].lower() in ("on", "1", "true", "yes")))
            elif c in ("next", "prev"):
                if not bridge.presets:
                    raise ValueError("no presets configured")
                q.put((c,))
            elif c == "preset" and len(t) >= 2:
                i = int(t[1])
                if not 1 <= i <= len(bridge.presets):
                    raise ValueError(f"preset must be 1..{len(bridge.presets)}")
                q.put(("preset", i - 1))
            elif c == "rotate" and len(t) >= 2:
                q.put(("rotate", 0.0 if t[1].lower() in ("off", "0") else float(t[1])))
            else:                                   # bare frequency, e.g. "145.5M"
                q.put(("freq", parse_freq(c)))
                if len(t) >= 2:
                    q.put(("mode", parse_mode(t[1])))
            replies.append("ok")
        except ValueError as e:
            replies.append(f"error: {e}")
    return " | ".join(replies) or "error: empty command"


class ControlServer(threading.Thread):
    def __init__(self, path: str, bridge):
        super().__init__(daemon=True)
        self.path, self.bridge = path, bridge
        if os.path.exists(path):
            os.unlink(path)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        os.chmod(path, 0o600)
        self.srv.listen(4)

    def run(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            conn.settimeout(30)
            with conn, conn.makefile("rw") as f:
                for line in f:
                    f.write(parse_command(line, self.bridge) + "\n")
                    f.flush()
        except OSError:
            pass

    def close(self):
        try:
            self.srv.close()
        finally:
            if os.path.exists(self.path):
                os.unlink(self.path)


def send_command(path: str, text: str) -> int:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect(path)
        s.sendall((text + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        print(s.makefile().read().strip())
        return 0
    except OSError as e:
        print(f"cannot reach {path}: {e}", file=sys.stderr)
        return 1
    finally:
        s.close()


# ----------------------------------------------------------------------------
# bridge
# ----------------------------------------------------------------------------
class Overlay:
    """Text files that ffmpeg's drawtext re-reads every frame (written atomically)."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="simtx_")
        self.main = os.path.join(self.dir, "main.txt")
        self.left = os.path.join(self.dir, "left.txt")
        self.right = os.path.join(self.dir, "right.txt")

    def _w(self, path, text):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    def update(self, freq, mode, span, note=""):
        txt = f"{freq / 1e6:.4f} MHz  {mode}"
        if note:
            txt += f"  |  {note}"
        self._w(self.main, txt)
        self._w(self.left, f"{(freq - span / 2) / 1e6:.3f} MHz")
        self._w(self.right, f"{(freq + span / 2) / 1e6:.3f} MHz")

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Bridge:
    def __init__(self, args, yt_key: str):
        self.args = args
        self.freq, self.mode, self.gain = args.freq, args.mode, args.gain
        self.presets = parse_presets(args.presets, args.mode)
        self.preset_idx = -1
        self.rotate = args.rotate if self.presets else 0.0
        self.next_rotate = time.monotonic() + self.rotate if self.rotate else None
        self.stop = False
        self.exit_code = 0
        self.next_try = 0.0
        self.status = "starting"
        self.cmd_q: "queue.Queue[tuple]" = queue.Queue()

        self.src = RtlSource(args.host, args.port)
        self.demod = Demod(self.mode)
        self.agc = DigitalAGC(target_level=0.4, alpha=0.003, max_gain=50.0) if args.agc else None
        self.spectrum = Spectrum(args.w, args.span)
        self.renderer = Renderer(args.w, args.h, args.db_range)
        self.overlay = Overlay()
        self.update_overlay()

        self.vpipe = None
        self.ff = self._start_ffmpeg(f"{args.rtmp_url}/{yt_key}")
        self.ctl = ControlServer(args.ctl_sock, self)
        self.ctl.start()

    # ---- ffmpeg ----------------------------------------------------------
    def _start_ffmpeg(self, rtmp_url: str) -> subprocess.Popen:
        a = self.args
        font = find_font(a.font)
        ff = f"fontfile={font}:" if font else ""
        if not font:
            log("WARNING: no TTF font found; relying on fontconfig (use --font if text is missing)")
        box = "fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"
        vf = ",".join([
            f"drawtext={ff}textfile={self.overlay.main}:reload=1:x=20:y=14:fontsize=34:{box}",
            f"drawtext={ff}text='{BRAND}':x=w-tw-20:y=14:fontsize=34:{box}",
            f"drawtext={ff}textfile={self.overlay.left}:reload=1:x=20:y=h-th-14:fontsize=22:{box}",
            f"drawtext={ff}textfile={self.overlay.right}:reload=1:x=w-tw-20:y=h-th-14:fontsize=22:{box}",
        ])
        vr, vw = os.pipe()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-thread_queue_size", "512", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{a.w}x{a.h}", "-framerate", str(a.fps), "-i", f"pipe:{vr}",
            "-thread_queue_size", "512", "-f", "s16le", "-ar", str(AUDIO_RATE), "-ac", "1", "-i", "pipe:0",
            "-map", "0:v", "-map", "1:a", "-vf", vf,
            "-c:v", "libx264", "-preset", a.x264_preset, "-pix_fmt", "yuv420p",
            "-r", str(a.fps), "-g", str(a.fps * 2), "-keyint_min", str(a.fps * 2), "-sc_threshold", "0",
            "-b:v", f"{a.bitrate}k", "-maxrate", f"{a.bitrate}k", "-bufsize", f"{a.bitrate * 2}k",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-f", "flv", rtmp_url,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, pass_fds=(vr,))
        os.close(vr)
        self.vpipe = os.fdopen(vw, "wb")
        return proc

    # ---- state / commands -----------------------------------------------
    def update_overlay(self):
        note = "" if self.src.sock else "NO SIGNAL - reconnecting"
        self.overlay.update(self.freq, self.mode, self.args.span, note)
        self.status = (f"freq={self.freq / 1e6:.6f}MHz mode={self.mode} gain={self.gain}dB "
                       f"agc={'on' if self.agc else 'off'} "
                       f"preset={self.preset_idx + 1 if self.preset_idx >= 0 else '-'}/{len(self.presets)} "
                       f"rotate={self.rotate or 'off'} rtl={'up' if self.src.sock else 'down'}")

    def _ds_value(self) -> int:
        d = self.args.direct_sampling
        if d == "auto":
            return 2 if self.freq < 24_000_000 else 0
        return {"off": 0, "i": 1, "q": 2}[d]

    def apply_gain(self):
        g = int(round(self.gain * 10))
        if g > 0:
            self.src.send(0x03, 1)
            self.src.send(0x04, g)
        else:
            self.src.send(0x03, 0)

    def configure_tuner(self):
        self.src.send(0x05, self.args.ppm)
        self.src.send(0x09, self._ds_value())
        self.src.send(0x01, self.freq)
        self.apply_gain()

    def _set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.demod = Demod(mode)

    def _goto_preset(self, i):
        self.preset_idx = i % len(self.presets)
        self.freq, mode = self.presets[self.preset_idx]
        self._set_mode(mode)

    def handle_commands(self):
        retune = False
        while True:
            try:
                c = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            k = c[0]
            if k == "freq":
                self.freq, retune = c[1], True
            elif k == "mode":
                self._set_mode(c[1])
                retune = True
            elif k == "gain":
                self.gain = c[1]
                if self.src.sock:
                    self.apply_gain()
            elif k == "agc":
                self.agc = DigitalAGC(target_level=0.4, alpha=0.003, max_gain=50.0) if c[1] else None
            elif k == "preset":
                self._goto_preset(c[1]); retune = True
            elif k in ("next", "prev"):
                self._goto_preset(self.preset_idx + (1 if k == "next" else -1)); retune = True
            elif k == "rotate":
                self.rotate = c[1]
                self.next_rotate = time.monotonic() + c[1] if c[1] and self.presets else None
        if retune:
            self.after_tune()
        else:
            self.update_overlay()

    def after_tune(self):
        self.demod.reset()
        if self.src.sock:
            self.src.send(0x09, self._ds_value())
            self.src.send(0x01, self.freq)
        if self.rotate:
            self.next_rotate = time.monotonic() + self.rotate
        self.update_overlay()
        log(f"tuned: {self.status}")

    def maybe_rotate(self, now):
        if self.next_rotate and self.presets and now >= self.next_rotate:
            self._goto_preset(self.preset_idx + 1)
            self.after_tune()

    # ---- IO ---------------------------------------------------------------
    def read_iq(self, now):
        if not self.src.sock:
            if now < self.next_try:
                return None
            try:
                self.src.connect()
                self.configure_tuner()
                log(f"connected to {self.args.host}:{self.args.port}")
                self.update_overlay()
            except OSError as e:
                log(f"connect failed: {e}")
                self.src.close()
                self.next_try = time.monotonic() + 3
                return None
        try:
            raw = self.src.read()
        except OSError as e:
            log(f"rtl_tcp lost ({e}); retrying")
            self.src.close()
            self.next_try = now + 2
            self.update_overlay()
            return None
        u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return ((u8[0::2] - 127.5) + 1j * (u8[1::2] - 127.5)).astype(np.complex64)

    def write(self, pcm: bytes, frames):
        self.ff.stdin.write(pcm)
        self.ff.stdin.flush()
        for fr in frames:
            self.vpipe.write(fr)
        self.vpipe.flush()

    # ---- main loop --------------------------------------------------------
    def run(self):
        fpc = self.args.fps // CHUNKS_PER_SEC
        next_t = time.monotonic()
        last_stat = time.monotonic()
        try:
            while not self.stop:
                now = time.monotonic()
                self.handle_commands()
                self.maybe_rotate(now)
                iq = self.read_iq(now)

                if iq is not None:
                    audio = self.demod.process(iq)
                    if self.agc:
                        audio = self.agc.process(audio)
                    seg = len(iq) // fpc
                    frames = [self.renderer.push(self.spectrum.row(iq[i * seg:(i + 1) * seg]))
                              for i in range(fpc)]
                    next_t = time.monotonic()
                else:                                       # keep the stream alive with silence
                    audio = np.zeros(AUDIO_PER_CHUNK, np.float32)
                    frames = [self.renderer.push(None) for _ in range(fpc)]
                    next_t += 1.0 / CHUNKS_PER_SEC
                    time.sleep(max(0.0, next_t - time.monotonic()))

                pcm = np.clip(audio, -32768, 32767).astype(np.int16).tobytes()
                self.write(pcm, frames)

                if time.monotonic() - last_stat > 60:
                    last_stat = time.monotonic()
                    log(f"alive: {self.status}")
        except (BrokenPipeError, ValueError) as e:
            log(f"ffmpeg pipe closed ({e}); exiting for supervisor restart")
            self.exit_code = 1
        finally:
            self.shutdown()

    def shutdown(self):
        self.src.close()
        self.ctl.close()
        for f in (self.ff.stdin, self.vpipe):
            try:
                f.close()
            except Exception:
                pass
        if self.ff.poll() is None:
            self.ff.terminate()
            try:
                self.ff.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ff.kill()
        self.overlay.close()


def main():
    args = get_args()

    if args.send:
        sys.exit(send_command(args.ctl_sock, args.send))

    yt_key = os.environ.get("YT_STREAM_KEY")
    if not yt_key:
        print("ERROR: YT_STREAM_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    log(f"SimTX starting: {args.host}:{args.port} | {args.freq / 1e6:.6f} MHz | {args.mode} | "
        f"gain {args.gain} dB | AGC {args.agc} | {args.w}x{args.h}@{args.fps}")
    bridge = Bridge(args, yt_key)

    def _stop(*_):
        bridge.stop = True
    pysignal.signal(pysignal.SIGTERM, _stop)
    pysignal.signal(pysignal.SIGINT, _stop)

    bridge.run()
    sys.exit(bridge.exit_code)


if __name__ == "__main__":
    main()
