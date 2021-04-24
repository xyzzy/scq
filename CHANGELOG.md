# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

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
