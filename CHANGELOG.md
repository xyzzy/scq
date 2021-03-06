# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

```
2021-05-03 16:10:32 Added: Reference to `bezeye-media`.
2021-05-01 22:44:20 Fixed: Freeze.
2021-04-30 22:29:09 Added: Timestamp prefix.
```

## 2021-04-30 13:19:50 [Version 0.4.0]

```
2021-04-30 13:15:08 Added: Save final palette with frequency count.
2021-04-30 12:54:26 Added: `--freeze`.
2021-04-30 11:18:54 Changed: Type `meanfield` from vector to array. 
2021-04-30 11:04:08 Changed: Better transparency handling.
2021-04-30 10:55:55 Changed: Replaced `double` with `float`.
2021-04-30 10:42:13 Changed: Normalize `image` by scaling with -0.5.  
2021-04-28 15:39:07 Changed: Replaced `a0` with `image`. `a0` is -2.0 of image.
2021-04-25 00:07:34 Changed: Replaced `b0` with `weights`. `b0` is double sized blur of weights.
2021-04-28 15:30:49 Fixed: Randomize visit queue. 
```

## 2021-04-27 23:05:32 [Version 0.3.0]

```
2021-04-27 23:02:49 Changed: Inlined `compute_initial_j_palette_sum` and adjusted `for` loop hierarchy.
2021-04-25 01:02:31 Changed: Replaced `weights` and `s` with single-double.
2021-04-27 22:01:15 Added: Save snapshots of internal image. 
```

## 2021-04-27 17:18:54 [Version 0.2.0]

```
2021-04-24 23:40:58 Changed: Redesigned visit queue.
2021-04-27 21:12:52 Fixed: Use `quantized_image` to detect changes and revisit pixels.
2021-04-24 23:40:58 Changed: Redesigned for loops.
2021-04-27 20:57:11 Fixed: Relocate palette construction to start of level loop.
2021-04-27 17:26:05 Removed: Coarse levels/zooming, possible because of octree palette prediction.
```

NOTE: Each of the above are minor changes that effect the outcome of the annealing.

## 2021-04-27 17:18:54 [Version 0.1.0]

```
2021-04-25 18:59:48 Fixed: shortcut to set identity matrix triggers a compiler bug.
2021-04-21 15:54:33 Added: `scq.cc` by replacing front-end.
2021-04-21 12:13:28 Added: Version used by other projects and original v0.4.
2021-04-05 03:14:49 Added: `moonwalk` theme.
2021-24-01 11:28:59 Historic commit.
```

[Unreleased]: https://github.com/xyzzy/scq/compare/v0.4.0...HEAD
[Version 0.4.0]: https://github.com/xyzzy/scq/compare/v0.3.0...v0.4.0
[Version 0.3.0]: https://github.com/xyzzy/scq/compare/v0.2.0...v0.3.0
[Version 0.2.0]: https://github.com/xyzzy/scq/compare/v0.1.0...v0.2.0
[Version 0.1.0]: https://github.com/xyzzy/scq/tree/v0.1.0