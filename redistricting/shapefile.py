"""Minimal ESRI shapefile / DBF reading -- no geometry library required.

Rings are returned as raw bytes (16 bytes per (x, y) vertex) rather than floats, so
two units that share a boundary can be matched by hashing identical byte pairs.
"""
import struct, math

R = 6371007.181          # GRS80 authalic radius (m), used for geodesic segment length


def read_dbf(p):
    f = open(p, "rb"); h = f.read(32)
    nrec, hlen, rlen = struct.unpack("<I H H", h[4:12])
    fl = []
    for _ in range((hlen - 33) // 32):
        d = f.read(32); fl.append((d[:11].rstrip(b"\0").decode(), d[16]))
    f.seek(hlen); out = []
    for _ in range(nrec):
        rec = f.read(rlen)
        if not rec or rec[:1] == b"\x1a": break
        o = 1; r = {}
        for nm, l in fl: r[nm] = rec[o:o+l].decode("latin-1").strip(); o += l
        out.append(r)
    return out


def read_shp(p):
    f = open(p, "rb"); f.seek(100); polys = []
    while True:
        h = f.read(8)
        if len(h) < 8: break
        _, clen = struct.unpack(">ii", h); body = f.read(clen * 2)
        if struct.unpack("<i", body[:4])[0] != 5:
            polys.append([]); continue
        npa, npt = struct.unpack("<ii", body[36:44])
        parts = struct.unpack(f"<{npa}i", body[44:44+4*npa]); co = 44 + 4*npa
        rings = []
        for i, s in enumerate(parts):
            e = parts[i+1] if i+1 < npa else npt
            rings.append(body[co+16*s : co+16*e])      # raw bytes, for hashing
        polys.append(rings)
    return polys


def seg_len(b1, b2):
    """Geodesic length of one segment, from two 16-byte (lon,lat) blobs."""
    x1, y1 = struct.unpack("<dd", b1); x2, y2 = struct.unpack("<dd", b2)
    p1, p2 = math.radians(y1), math.radians(y2)
    dl = math.radians(x2 - x1); dp = p2 - p1
    q = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(min(1.0, max(0.0, q))))
