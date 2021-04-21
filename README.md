---
layout: home
title: "Spatial Colour Quantization - True-colour image to palette conversion"
image: assets/favimage-840x472.jpg
---

# scq

Spatial Colour Quantization, True-colour images to palette conversion

### Welcome to the Wonderful World of colour compression

Spatial Colour Quantization is a lossy colour compression.  
It introduces dithering as background noise, allowing the mixing of colours in a natural seeming way.  
With a limited and well-chosen colours, it can convincingly reduce 24-bit RGB to less than 6 bits palette.

This version of `scq` has evolved into handling video and transparency.  
The latest collaboration is with the [splash](https://rockingship.github.io/splash/README.html)` codec.

`splash` prioritizes the position of pixels and `scq` prioritizes the RGB value of pixels.

The original research paper can be found here:  
  [https://www.researchgate.net/publication/220502178_On_spatial_quantization_of_color_images](https://www.researchgate.net/publication/220502178_On_spatial_quantization_of_color_images)

The page with D. Coetzee's implementation:  
  [https://people.eecs.berkeley.edu/~dcoetzee/downloads/scolorq](https://people.eecs.berkeley.edu/~dcoetzee/downloads/scolorq)

## Manifest

 - [bezeye-scq.cc](bezeye-scq.cc)  
   Adaptation used for [bezeye](https://xyzzy.github.io/bezeye/README.html), focuses on animation/transparency.

 - [gaia1-scq.cc](gaia1-scq.cc)  
   Adaptation used for [gaia1](https://xyzzy.github.io/gaia1/README.html), focuses on animation/transparency/size.  
   Gaia1 was submitted as part of the Revision 2014 Animated Gif Compo.

 - [qrpicture-scq.cc](qrpicture-scq.cc)  
   Adaptation used for [qrpicture](https://xyzzy.github.io/qrpicture/README.html), focuses on creating QR safe colour ranges.

 - [scolorq-0.4.cc](scolorq-0.4.cc)  
   Original v0.4 implementation by D. Coetzee.

## Source code

Grab one of the tarballs at [https://github.com/xyzzy/scq/releases](https://github.com/xyzzy/scq/releases) or checkout the latest code:

```sh
  git clone https://github.com/xyzzy/scq.git
```

## Versioning

Using [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/xyzzy/scq/tags).

## License

This project is licensed under Affero GPLv3 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

 - D. Coetzee and his amazing work on the mathematical background of Spatial Colour Quantization