"""Pure-Python image decoding and encoding.

FruitVision ships with zero third-party dependencies, so this module replaces
Pillow.  It provides just enough of the PNG and JPEG formats to load the
training set, accept user uploads and write preview images:

* ``decode_png``  -- non-interlaced PNG, bit depths 1/2/4/8/16,
  grayscale / RGB / palette / alpha.
* ``decode_jpeg`` -- baseline sequential JPEG with Huffman coding and
  arbitrary chroma sub-sampling.
* ``decode_bmp``  -- uncompressed 24/32-bit BMP.
* ``encode_png``  -- 8-bit RGB PNG writer (used for dataset preparation).

Everything is expressed with the standard library only (``zlib``, ``struct``).
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Optional, Tuple

__all__ = [
    "Image",
    "UnsupportedImageError",
    "decode_image",
    "decode_png",
    "probe_jpeg_size",
    "decode_jpeg",
    "decode_bmp",
    "encode_png",
    "sniff_format",
]


class UnsupportedImageError(ValueError):
    """Raised when a file is not an image we know how to decode."""


class Image:
    """A decoded RGB image.

    Pixels are stored as a flat :class:`bytearray` of interleaved R, G, B
    bytes, ``3 * width * height`` long.  This is deliberately the same layout
    the browser's ``canvas.getImageData`` produces (minus alpha), which keeps
    the training and serving paths byte-for-byte consistent.
    """

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, pixels: bytearray) -> None:
        expected = width * height * 3
        if len(pixels) != expected:
            raise ValueError(
                "pixel buffer is %d bytes, expected %d for %dx%d RGB"
                % (len(pixels), expected, width, height)
            )
        self.width = width
        self.height = height
        self.pixels = pixels

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def getpixel(self, x: int, y: int) -> Tuple[int, int, int]:
        i = (y * self.width + x) * 3
        return (self.pixels[i], self.pixels[i + 1], self.pixels[i + 2])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<Image %dx%d RGB>" % (self.width, self.height)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sniff_format(data: bytes) -> str:
    """Return ``'png'``, ``'jpeg'``, ``'bmp'`` or ``'unknown'``."""

    if data[:8] == _PNG_MAGIC:
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:2] == b"BM":
        return "bmp"
    return "unknown"


def probe_jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Read a JPEG's dimensions from its frame header without decoding it.

    Used to decide whether the 1/8-scale fast path still leaves enough detail.
    """

    pos = 2
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        pos += 2
        if marker == 0xFF:
            pos -= 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9 or marker == 0xDA:
            return None
        if pos + 2 > len(data):
            return None
        (length,) = struct.unpack(">H", data[pos : pos + 2])
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[pos + 3 : pos + 7])
            return (width, height)
        pos += length
    return None


# Below this size the 1/8-scale decode would leave too little detail, so the
# image is decoded in full instead (which is quick for a small file anyway).
FAST_DECODE_MIN_EDGE = 256


def decode_image(data: bytes, fast: bool = False) -> Image:
    """Decode any supported image format into an :class:`Image`.

    ``fast`` asks the JPEG path for a 1/8-scale decode, which is dramatically
    quicker on multi-megapixel photos and is plenty of detail for a classifier
    whose input is only a few dozen pixels across.
    """

    fmt = sniff_format(data)
    if fmt == "png":
        return decode_png(data)
    if fmt == "jpeg":
        dc_only = fast
        if dc_only:
            size = probe_jpeg_size(data)
            if size is not None and min(size) < FAST_DECODE_MIN_EDGE:
                dc_only = False
        return decode_jpeg(data, dc_only=dc_only)
    if fmt == "bmp":
        return decode_bmp(data)
    raise UnsupportedImageError(
        "Unrecognised image format. Supported formats are PNG, JPEG and BMP."
    )


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


def _png_unfilter(raw: bytes, width: int, height: int, bpp: int, stride: int) -> bytearray:
    """Reverse the per-scanline PNG filters (RFC 2083 section 6)."""

    out = bytearray(stride * height)
    pos = 0
    prev_start = 0
    for row in range(height):
        filter_type = raw[pos]
        pos += 1
        start = row * stride
        out[start : start + stride] = raw[pos : pos + stride]
        pos += stride

        if filter_type == 0:  # None
            pass
        elif filter_type == 1:  # Sub
            for i in range(bpp, stride):
                out[start + i] = (out[start + i] + out[start + i - bpp]) & 0xFF
        elif filter_type == 2:  # Up
            if row:
                for i in range(stride):
                    out[start + i] = (out[start + i] + out[prev_start + i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = out[start + i - bpp] if i >= bpp else 0
                up = out[prev_start + i] if row else 0
                out[start + i] = (out[start + i] + ((left + up) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = out[start + i - bpp] if i >= bpp else 0
                up = out[prev_start + i] if row else 0
                upleft = out[prev_start + i - bpp] if (row and i >= bpp) else 0
                p = left + up - upleft
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - upleft)
                if pa <= pb and pa <= pc:
                    pred = left
                elif pb <= pc:
                    pred = up
                else:
                    pred = upleft
                out[start + i] = (out[start + i] + pred) & 0xFF
        else:
            raise UnsupportedImageError("Unknown PNG filter type %d" % filter_type)
        prev_start = start
    return out


def _unpack_bits(row: bytes, depth: int, count: int) -> List[int]:
    """Expand sub-byte samples (depth 1/2/4) into one value per sample."""

    values: List[int] = []
    per_byte = 8 // depth
    mask = (1 << depth) - 1
    for byte in row:
        for k in range(per_byte):
            shift = 8 - depth * (k + 1)
            values.append((byte >> shift) & mask)
            if len(values) == count:
                return values
    return values


def decode_png(data: bytes) -> Image:
    """Decode a non-interlaced PNG into RGB, compositing alpha over white."""

    if data[:8] != _PNG_MAGIC:
        raise UnsupportedImageError("Not a PNG file")

    pos = 8
    width = height = 0
    depth = color_type = interlace = 0
    palette = b""
    trns: Optional[bytes] = None
    idat = bytearray()

    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + body + 4 CRC

        if ctype == b"IHDR":
            width, height, depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body
            )
        elif ctype == b"PLTE":
            palette = body
        elif ctype == b"tRNS":
            trns = body
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break

    if not width or not height:
        raise UnsupportedImageError("PNG is missing its IHDR header")
    if interlace:
        raise UnsupportedImageError(
            "Interlaced (Adam7) PNGs are not supported. Re-save the image without interlacing."
        )
    if depth not in (1, 2, 4, 8, 16):
        raise UnsupportedImageError("Unsupported PNG bit depth %d" % depth)

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise UnsupportedImageError("Unsupported PNG colour type %d" % color_type)

    raw = zlib.decompress(bytes(idat))
    bits_per_pixel = channels * depth
    stride = (width * bits_per_pixel + 7) // 8
    bpp = max(1, bits_per_pixel // 8)
    unfiltered = _png_unfilter(raw, width, height, bpp, stride)

    pixels = bytearray(width * height * 3)
    out = 0
    smax = (1 << depth) - 1

    for row in range(height):
        line = unfiltered[row * stride : (row + 1) * stride]

        if depth == 16:
            samples = [line[i] for i in range(0, len(line), 2)]  # keep high byte
        elif depth == 8:
            samples = list(line)
        else:
            samples = _unpack_bits(bytes(line), depth, width * channels)

        for col in range(width):
            base = col * channels
            if color_type == 0:  # grayscale
                g = samples[base]
                if depth not in (8, 16):
                    g = g * 255 // smax
                r = gg = b = g
                a = 255
            elif color_type == 2:  # RGB
                r, gg, b = samples[base], samples[base + 1], samples[base + 2]
                a = 255
            elif color_type == 3:  # palette
                idx = samples[base]
                r = palette[idx * 3]
                gg = palette[idx * 3 + 1]
                b = palette[idx * 3 + 2]
                a = trns[idx] if (trns is not None and idx < len(trns)) else 255
            elif color_type == 4:  # grayscale + alpha
                g = samples[base]
                a = samples[base + 1]
                if depth not in (8, 16):
                    g = g * 255 // smax
                    a = a * 255 // smax
                r = gg = b = g
            else:  # 6 == RGBA
                r, gg, b = samples[base], samples[base + 1], samples[base + 2]
                a = samples[base + 3]

            if a != 255:  # composite over a white background
                inv = 255 - a
                r = (r * a + 255 * inv) // 255
                gg = (gg * a + 255 * inv) // 255
                b = (b * a + 255 * inv) // 255

            pixels[out] = r
            pixels[out + 1] = gg
            pixels[out + 2] = b
            out += 3

    return Image(width, height, pixels)


def encode_png(image: Image, compress_level: int = 9) -> bytes:
    """Encode an :class:`Image` as an 8-bit RGB PNG."""

    stride = image.width * 3
    raw = bytearray()
    for row in range(image.height):
        raw.append(0)  # filter type: None
        raw += image.pixels[row * stride : (row + 1) * stride]

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    return (
        _PNG_MAGIC
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# BMP
# ---------------------------------------------------------------------------


def decode_bmp(data: bytes) -> Image:
    """Decode an uncompressed 24- or 32-bit BMP."""

    if data[:2] != b"BM":
        raise UnsupportedImageError("Not a BMP file")

    (offset,) = struct.unpack("<I", data[10:14])
    (header_size,) = struct.unpack("<I", data[14:18])
    if header_size < 40:
        raise UnsupportedImageError("Unsupported BMP header size %d" % header_size)

    width, height = struct.unpack("<ii", data[18:26])
    bpp, compression = struct.unpack("<HI", data[28:34])
    if compression != 0 or bpp not in (24, 32):
        raise UnsupportedImageError(
            "Only uncompressed 24/32-bit BMP files are supported."
        )

    bottom_up = height > 0
    height = abs(height)
    sample_bytes = bpp // 8
    stride = ((width * sample_bytes + 3) // 4) * 4

    pixels = bytearray(width * height * 3)
    for row in range(height):
        src_row = (height - 1 - row) if bottom_up else row
        start = offset + src_row * stride
        for col in range(width):
            i = start + col * sample_bytes
            b, g, r = data[i], data[i + 1], data[i + 2]
            o = (row * width + col) * 3
            pixels[o] = r
            pixels[o + 1] = g
            pixels[o + 2] = b
    return Image(width, height, pixels)


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------

# Order in which the 64 coefficients of a block are stored in the bitstream.
_ZIGZAG = (
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)


def _build_idct_table() -> List[List[float]]:
    """Pre-compute the separable 8-point IDCT basis."""

    import math

    table = []
    for u in range(8):
        scale = (0.5 / math.sqrt(2.0)) if u == 0 else 0.5
        table.append([scale * math.cos((2 * x + 1) * u * math.pi / 16.0) for x in range(8)])
    return table


_IDCT = _build_idct_table()

# Clamp lookup: index by (value + 256) to convert IDCT output to a byte.
_CLAMP = bytes(min(255, max(0, i - 256)) for i in range(768))


class _Huffman:
    """A JPEG Huffman table stored as a ``{(bit_length, code): value}`` map."""

    __slots__ = ("lookup", "max_length")

    def __init__(self, counts: bytes, symbols: bytes) -> None:
        self.lookup = {}
        code = 0
        k = 0
        self.max_length = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                self.lookup[(length, code)] = symbols[k]
                code += 1
                k += 1
            code <<= 1
            if counts[length - 1]:
                self.max_length = length


class _BitReader:
    """Reads entropy-coded JPEG data, undoing 0xFF00 byte stuffing."""

    __slots__ = ("data", "pos", "buf", "count", "eof")

    def __init__(self, data: bytes, pos: int) -> None:
        self.data = data
        self.pos = pos
        self.buf = 0
        self.count = 0
        self.eof = False

    def read_bit(self) -> int:
        if self.count == 0:
            if self.pos >= len(self.data):
                self.eof = True
                return 0
            byte = self.data[self.pos]
            self.pos += 1
            if byte == 0xFF:
                nxt = self.data[self.pos] if self.pos < len(self.data) else 0xD9
                if nxt == 0x00:
                    self.pos += 1  # stuffed byte, 0xFF is real data
                else:
                    # A real marker: the entropy-coded segment ends here.
                    self.pos -= 1
                    self.eof = True
                    return 0
            self.buf = byte
            self.count = 8
        self.count -= 1
        return (self.buf >> self.count) & 1

    def receive(self, length: int) -> int:
        value = 0
        for _ in range(length):
            value = (value << 1) | self.read_bit()
        return value

    def decode(self, table: _Huffman) -> int:
        code = 0
        lookup = table.lookup
        for length in range(1, 17):
            code = (code << 1) | self.read_bit()
            value = lookup.get((length, code))
            if value is not None:
                return value
        raise UnsupportedImageError("Corrupt JPEG: invalid Huffman code")

    def restart(self) -> None:
        """Align to a byte boundary and consume an RSTn marker if present."""

        self.count = 0
        self.eof = False
        while self.pos + 1 < len(self.data):
            if self.data[self.pos] == 0xFF and 0xD0 <= self.data[self.pos + 1] <= 0xD7:
                self.pos += 2
                return
            if self.data[self.pos] == 0xFF and self.data[self.pos + 1] != 0x00:
                return  # a different marker: let the caller handle it
            self.pos += 1


def _extend(value: int, length: int) -> int:
    """Convert a JPEG magnitude-category value into a signed integer."""

    if length == 0:
        return 0
    return value if value >= (1 << (length - 1)) else value - (1 << length) + 1


class _Component:
    __slots__ = (
        "cid", "h", "v", "tq", "dc_table", "ac_table", "pred",
        "blocks_per_line", "blocks_per_column", "blocks", "eobrun",
    )


def _decode_scan(
    reader: _BitReader,
    components: List["_Component"],
    mcus_per_line: int,
    mcus_per_column: int,
    restart_interval: int,
    spectral_start: int,
    spectral_end: int,
    approx_high: int,
    approx_low: int,
    progressive: bool,
) -> None:
    """Decode one scan, accumulating into each component's coefficient blocks."""

    state = {"eobrun": 0}

    def decode_baseline(comp: "_Component", block: List[int]) -> None:
        t = reader.decode(comp.dc_table)
        diff = _extend(reader.receive(t), t) if t else 0
        comp.pred += diff
        block[0] = comp.pred
        k = 1
        while k < 64:
            rs = reader.decode(comp.ac_table)
            run, size = rs >> 4, rs & 15
            if size == 0:
                if run != 15:
                    break
                k += 16
                continue
            k += run
            if k > 63:
                break
            block[_ZIGZAG[k]] = _extend(reader.receive(size), size)
            k += 1

    def decode_dc_first(comp: "_Component", block: List[int]) -> None:
        t = reader.decode(comp.dc_table)
        diff = _extend(reader.receive(t), t) if t else 0
        comp.pred += diff
        block[0] = comp.pred << approx_low

    def decode_dc_refine(comp: "_Component", block: List[int]) -> None:
        if reader.read_bit():
            block[0] |= 1 << approx_low

    def decode_ac_first(comp: "_Component", block: List[int]) -> None:
        if state["eobrun"] > 0:
            state["eobrun"] -= 1
            return
        k = spectral_start
        while k <= spectral_end:
            rs = reader.decode(comp.ac_table)
            run, size = rs >> 4, rs & 15
            if size == 0:
                if run < 15:
                    state["eobrun"] = (1 << run) - 1
                    if run:
                        state["eobrun"] += reader.receive(run)
                    break
                k += 16
                continue
            k += run
            if k > 63:
                break
            block[_ZIGZAG[k]] = _extend(reader.receive(size), size) << approx_low
            k += 1

    def decode_ac_refine(comp: "_Component", block: List[int]) -> None:
        k = spectral_start
        plus_one = 1 << approx_low
        minus_one = -1 << approx_low

        if state["eobrun"] <= 0:
            while k <= spectral_end:
                rs = reader.decode(comp.ac_table)
                run, size = rs >> 4, rs & 15
                if size == 0:
                    if run < 15:
                        state["eobrun"] = (1 << run)
                        if run:
                            state["eobrun"] += reader.receive(run)
                        break
                    # run == 15: skip 16 zero-history coefficients
                    value = 0
                else:
                    value = plus_one if reader.read_bit() else minus_one

                while k <= spectral_end:
                    z = _ZIGZAG[k]
                    if block[z] != 0:
                        if reader.read_bit() and (block[z] & plus_one) == 0:
                            block[z] += plus_one if block[z] >= 0 else minus_one
                    else:
                        if run == 0:
                            if size:
                                block[z] = value
                            k += 1
                            break
                        run -= 1
                    k += 1
                else:
                    break

        if state["eobrun"] > 0:
            # Correction bits for the remaining non-zero coefficients.
            while k <= spectral_end:
                z = _ZIGZAG[k]
                if block[z] != 0:
                    if reader.read_bit() and (block[z] & plus_one) == 0:
                        block[z] += plus_one if block[z] >= 0 else minus_one
                k += 1
            state["eobrun"] -= 1

    if not progressive:
        decode_block = decode_baseline
    elif spectral_start == 0:
        decode_block = decode_dc_refine if approx_high else decode_dc_first
    else:
        decode_block = decode_ac_refine if approx_high else decode_ac_first

    for comp in components:
        comp.pred = 0
    state["eobrun"] = 0

    single = len(components) == 1
    if single:
        comp = components[0]
        total_units = comp.blocks_per_line * comp.blocks_per_column
    else:
        total_units = mcus_per_line * mcus_per_column

    interval = restart_interval or total_units
    unit = 0
    while unit < total_units:
        stop = min(total_units, unit + interval)
        while unit < stop:
            if single:
                row, col = divmod(unit, comp.blocks_per_line)
                decode_block(comp, comp.blocks[row][col])
            else:
                mcu_row, mcu_col = divmod(unit, mcus_per_line)
                for c in components:
                    for vy in range(c.v):
                        for hx in range(c.h):
                            decode_block(
                                c, c.blocks[mcu_row * c.v + vy][mcu_col * c.h + hx]
                            )
            unit += 1
        if unit < total_units:
            reader.restart()
            for c in components:
                c.pred = 0
            state["eobrun"] = 0


def _idct_block(coeffs: List[int], quant: List[int], out: bytearray) -> None:
    """Dequantise and inverse-transform one 8x8 block into 64 clamped bytes."""

    if not any(coeffs[1:]):  # DC-only block: the result is a flat tone
        value = _CLAMP[int(coeffs[0] * quant[0] * 0.125 + 128.5) + 256]
        for i in range(64):
            out[i] = value
        return

    dequant = [coeffs[i] * quant[i] for i in range(64)]

    # Pass 1: rows.
    tmp = [0.0] * 64
    for y in range(8):
        base = y * 8
        row = dequant[base : base + 8]
        if not any(row[1:]):
            value = row[0] * 0.35355339059327373
            for x in range(8):
                tmp[base + x] = value
            continue
        for x in range(8):
            total = 0.0
            for u in range(8):
                if row[u]:
                    total += _IDCT[u][x] * row[u]
            tmp[base + x] = total

    # Pass 2: columns.
    for x in range(8):
        col = [tmp[y * 8 + x] for y in range(8)]
        for y in range(8):
            total = 0.0
            for v in range(8):
                if col[v]:
                    total += _IDCT[v][y] * col[v]
            out[y * 8 + x] = _CLAMP[int(total + 128.5) + 256]


def _ycbcr_to_rgb(y: int, cb: int, cr: int) -> Tuple[int, int, int]:
    cb -= 128
    cr -= 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return (
        _CLAMP[int(r + 0.5) + 256],
        _CLAMP[int(g + 0.5) + 256],
        _CLAMP[int(b + 0.5) + 256],
    )


def decode_jpeg(data: bytes, dc_only: bool = False) -> Image:
    """Decode a baseline or progressive JPEG.

    When ``dc_only`` is true only the DC coefficient of each 8x8 block is used,
    producing an image at 1/8 scale.  Each output pixel is then the average of
    its source block, which is exactly what we want before downscaling to the
    model's input size -- and it skips the expensive inverse DCT entirely.
    """

    if data[:2] != b"\xff\xd8":
        raise UnsupportedImageError("Not a JPEG file")

    quant_tables = {}
    dc_tables = {}
    ac_tables = {}
    components: List[_Component] = []
    frame_width = frame_height = 0
    progressive = False
    restart_interval = 0
    max_h = max_v = 1
    mcus_per_line = mcus_per_column = 0

    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        pos += 2

        if marker == 0xFF:
            pos -= 1  # 0xFF fill bytes may pad the run-in to a marker
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:  # EOI
            break

        (length,) = struct.unpack(">H", data[pos : pos + 2])
        segment = data[pos + 2 : pos + length]

        if marker == 0xDB:  # DQT
            i = 0
            while i < len(segment):
                precision, table_id = segment[i] >> 4, segment[i] & 15
                i += 1
                table = [0] * 64
                for k in range(64):
                    if precision:
                        (value,) = struct.unpack(">H", segment[i : i + 2])
                        i += 2
                    else:
                        value = segment[i]
                        i += 1
                    table[_ZIGZAG[k]] = value
                quant_tables[table_id] = table

        elif marker in (0xC0, 0xC1, 0xC2):  # SOF0 / SOF1 / SOF2
            progressive = marker == 0xC2
            frame_height, frame_width = struct.unpack(">HH", segment[1:5])
            count = segment[5]
            components = []
            for i in range(count):
                off = 6 + i * 3
                comp = _Component()
                comp.cid = segment[off]
                comp.h = segment[off + 1] >> 4
                comp.v = segment[off + 1] & 15
                comp.tq = segment[off + 2]
                comp.pred = 0
                comp.eobrun = 0
                components.append(comp)

            max_h = max(c.h for c in components)
            max_v = max(c.v for c in components)
            mcus_per_line = -(-frame_width // (8 * max_h))
            mcus_per_column = -(-frame_height // (8 * max_v))
            for comp in components:
                comp.blocks_per_line = -(-(-(-frame_width // 8) * comp.h) // max_h)
                comp.blocks_per_column = -(-(-(-frame_height // 8) * comp.v) // max_v)
                rows = mcus_per_column * comp.v
                cols = mcus_per_line * comp.h
                comp.blocks = [
                    [[0] * 64 for _ in range(cols)] for _ in range(rows)
                ]

        elif marker in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise UnsupportedImageError(
                "Arithmetic-coded, hierarchical or lossless JPEGs are not supported."
            )

        elif marker == 0xC4:  # DHT
            i = 0
            while i < len(segment):
                table_class, table_id = segment[i] >> 4, segment[i] & 15
                counts = segment[i + 1 : i + 17]
                total = sum(counts)
                symbols = segment[i + 17 : i + 17 + total]
                table = _Huffman(counts, symbols)
                if table_class == 0:
                    dc_tables[table_id] = table
                else:
                    ac_tables[table_id] = table
                i += 17 + total

        elif marker == 0xDD:  # DRI
            (restart_interval,) = struct.unpack(">H", segment[:2])

        elif marker == 0xDA:  # SOS
            count = segment[0]
            scan_components = []
            for i in range(count):
                cid = segment[1 + i * 2]
                tables = segment[2 + i * 2]
                comp = next((c for c in components if c.cid == cid), None)
                if comp is None:
                    raise UnsupportedImageError("Corrupt JPEG: unknown scan component")
                comp.dc_table = dc_tables.get(tables >> 4)
                comp.ac_table = ac_tables.get(tables & 15)
                scan_components.append(comp)

            off = 1 + count * 2
            spectral_start = segment[off]
            spectral_end = segment[off + 1]
            approx_high = segment[off + 2] >> 4
            approx_low = segment[off + 2] & 15

            reader = _BitReader(data, pos + length)
            _decode_scan(
                reader, scan_components, mcus_per_line, mcus_per_column,
                restart_interval, spectral_start, spectral_end,
                approx_high, approx_low, progressive,
            )
            pos = reader.pos
            continue

        pos += length

    if not components or not frame_width or not frame_height:
        raise UnsupportedImageError("Corrupt JPEG: no frame header found")

    return _render_jpeg(
        components, quant_tables, frame_width, frame_height, max_h, max_v, dc_only
    )


def _render_jpeg(components, quant_tables, width, height, max_h, max_v, dc_only):
    """Turn decoded coefficient blocks into an RGB :class:`Image`."""

    if dc_only:
        out_w = max(1, -(-width // 8))
        out_h = max(1, -(-height // 8))
    else:
        out_w, out_h = width, height

    planes = []
    for comp in components:
        quant = quant_tables.get(comp.tq, [1] * 64)
        rows = len(comp.blocks)
        cols = len(comp.blocks[0]) if rows else 0

        if dc_only:
            plane_w, plane_h = cols, rows
            plane = bytearray(plane_w * plane_h)
            scale = quant[0] * 0.125
            for by in range(rows):
                row_blocks = comp.blocks[by]
                base = by * plane_w
                for bx in range(cols):
                    plane[base + bx] = _CLAMP[
                        int(row_blocks[bx][0] * scale + 128.5) + 256
                    ]
        else:
            plane_w, plane_h = cols * 8, rows * 8
            plane = bytearray(plane_w * plane_h)
            pixels8 = bytearray(64)
            for by in range(rows):
                row_blocks = comp.blocks[by]
                for bx in range(cols):
                    _idct_block(row_blocks[bx], quant, pixels8)
                    for y in range(8):
                        start = (by * 8 + y) * plane_w + bx * 8
                        plane[start : start + 8] = pixels8[y * 8 : y * 8 + 8]

        # How many output pixels one sample of this plane covers.
        planes.append((plane, plane_w, plane_h, comp.h, comp.v))

    pixels = bytearray(out_w * out_h * 3)
    grayscale = len(planes) == 1
    out = 0

    for y in range(out_h):
        for x in range(out_w):
            samples = []
            for plane, plane_w, plane_h, ch, cv in planes:
                # Map output coordinates onto this (possibly sub-sampled) plane.
                sx = x * ch // max_h
                sy = y * cv // max_v
                if sx >= plane_w:
                    sx = plane_w - 1
                if sy >= plane_h:
                    sy = plane_h - 1
                samples.append(plane[sy * plane_w + sx])

            if grayscale:
                g = samples[0]
                r, gg, b = g, g, g
            else:
                r, gg, b = _ycbcr_to_rgb(samples[0], samples[1], samples[2])
            pixels[out] = r
            pixels[out + 1] = gg
            pixels[out + 2] = b
            out += 3

    return Image(out_w, out_h, pixels)
