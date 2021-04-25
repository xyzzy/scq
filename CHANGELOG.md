# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

NOTE: Pixel detection change: `if ((palette[max_v] - palette[old_max_v]).norm_squared() >= 1.0 / (255.0 * 255.0))`
      Is replaced with: `if (max_v != old_max_v)` because this indicates an actual change in palette index.

```
2021-04-25 01:13:50 Changed: Simplify pixel change detection.
2021-04-25 01:03:56 Changed: `s` is only used for palette updates, relocated everything there. 
2021-04-25 01:02:31 Changed: Replaced `weights` and `s` with single-double.
```

NOTE: `b0` is basically an upscaled and larger grid than `filter_weights`.
      Dropping `b0` in favour for weights is an enormous performance boost and might require re-tuning.

```
2021-04-25 00:07:34 Changed: Replaced `b0` with `weights`.
```

NOTE: Apparently it is more effective to jump directly to the final temperature and loop until stable.  
      Possible because of initial octree approximation.

```
2021-04-24 23:40:58 Changed: Redesigned visit queue.
2021-04-24 23:20:52 Changed: Move allocation of `variables` to caller.
```

NOTE: By chance of coincidence the original code worked with the visit queue.
      The queue is reset after 10% expansion visiting all pixels, effectively greatly increasing repeatPerLevel.  
      This needs to be re-tunes.

```
2021-04-24 22:31:02 Changed: Set `max_coarse_level=0` and drop coarse levels (because octree gives sufficient accuracy). 
2021-04-21 15:54:33 Added: `scq.cc` by replacing front-end.
2021-04-21 12:13:28 Added: Version used by other projects and original v0.4.
2021-04-05 03:14:49 Added: `moonwalk` theme.
2021-24-01 11:28:59 Historic commit.
```
