/**
 * Streamed, atomic file writes and line iteration over a Buffer.
 *
 * Every graft artifact used to be one `JSON.stringify` and one `JSON.parse`.
 * V8 caps a string at ~512 MiB (`buffer.constants.MAX_STRING_LENGTH`); a
 * 65k-file repo's extract cache is ~1 GB and its wiring.json ~0.5 GB, so the
 * cold build threw `Invalid string length` at the very end, and a graph that
 * did get written threw `ERR_STRING_TOO_LONG` on read. The writers here take
 * one element at a time; the reader decodes one line at a time.
 */
import { closeSync, mkdirSync, openSync, renameSync, rmSync, writeSync } from "node:fs";
import { dirname } from "node:path";

export interface AtomicWriter {
  write(chunk: string): void;
  commit(): void;
  abort(): void;
}

/** Write to `<path>.<pid>.tmp`, then rename over `path` on commit(). The pid keeps
 * two concurrent writers off each other's scratch file, the same discipline as
 * `writeJsonAtomic` in util/state.ts. */
export function openAtomic(path: string): AtomicWriter {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.tmp`;
  let fd: number | null = openSync(tmp, "w");
  const cleanup = (): void => {
    if (fd !== null) { try { closeSync(fd); } catch { /* already closed */ } fd = null; }
    try { rmSync(tmp, { force: true }); } catch { /* nothing more to do */ }
  };
  return {
    write(chunk) {
      if (fd === null) throw new Error("atomic writer is closed");
      // writeSync may write fewer bytes than asked (pipes, some filesystems):
      // loop on the byte count, so a multi-megabyte element is never truncated.
      const bytes = Buffer.from(chunk, "utf8");
      let off = 0;
      while (off < bytes.length) off += writeSync(fd, bytes, off, bytes.length - off);
    },
    commit() {
      if (fd === null) throw new Error("atomic writer is closed");
      try {
        closeSync(fd);
        fd = null;
        renameSync(tmp, path);
      } catch (e) {
        cleanup();
        throw e;
      }
    },
    abort() { cleanup(); },
  };
}

/** Call `fn` for each line of `buf`. A trailing line without "\n" is still a line;
 * an empty buffer yields none. Each line is decoded on its own, so the Buffer may
 * be larger than any string V8 can hold. */
export function forEachLine(buf: Buffer, fn: (line: string, index: number) => void): void {
  let start = 0;
  let index = 0;
  while (start < buf.length) {
    let end = buf.indexOf(0x0a, start);
    if (end === -1) end = buf.length;
    fn(buf.toString("utf8", start, end), index++);
    start = end + 1;
  }
}
