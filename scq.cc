/*
 * This file is part of scq, True-colour image to palette conversion
 * Copyright (C) 2021, xyzzy@rockingship.net
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Based on the original work of:
 *  Copyright (c) 2006 Derrick Coetzee
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <vector>
#include <deque>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <limits>
#include <getopt.h>
#include <stdint.h>
#include <string.h>
#include <gd.h>

#define MAXPALETTE 256

using namespace std;

template<typename T, int length>
class vector_fixed {
public:
	vector_fixed() {
		for (int i = 0; i < length; i++) {
			data[i] = 0;
		}
	}

	vector_fixed(const vector_fixed<T, length> &rhs) {
		for (int i = 0; i < length; i++) {
			data[i] = rhs.data[i];
		}
	}

	vector_fixed(const vector<T> &rhs) {
		for (int i = 0; i < length; i++) {
			data[i] = rhs[i];
		}
	}

	T &operator()(int i) {
		return data[i];
	}

	vector_fixed<T, length> &operator=(const vector_fixed<T, length> rhs) {
		for (int i = 0; i < length; i++) {
			data[i] = rhs.data[i];
		}
		return *this;
	}

	float dot_product(vector_fixed<T, length> rhs) {
		T result = 0;
		for (int i = 0; i < length; i++) {
			result += (*this)(i) * rhs(i);
		}
		return result;
	}

	vector_fixed<T, length> &operator+=(vector_fixed<T, length> rhs) {
		for (int i = 0; i < length; i++) {
			data[i] += rhs(i);
		}
		return *this;
	}

	vector_fixed<T, length> operator+(vector_fixed<T, length> rhs) {
		vector_fixed<T, length> result(*this);
		result += rhs;
		return result;
	}

	vector_fixed<T, length> &operator-=(vector_fixed<T, length> rhs) {
		for (int i = 0; i < length; i++) {
			data[i] -= rhs(i);
		}
		return *this;
	}

	vector_fixed<T, length> operator-(vector_fixed<T, length> rhs) {
		vector_fixed<T, length> result(*this);
		result -= rhs;
		return result;
	}

	vector_fixed<T, length> &operator*=(T scalar) {
		for (int i = 0; i < length; i++) {
			data[i] *= scalar;
		}
		return *this;
	}

	vector_fixed<T, length> operator*(T scalar) {
		vector_fixed<T, length> result(*this);
		result *= scalar;
		return result;
	}

private:
	T data[length];
};

template<typename T, int length>
vector_fixed<T, length> operator*(T scalar, vector_fixed<T, length> vec) {
	return vec * scalar;
}


template<typename T, int length>
ostream &operator<<(ostream &out, vector_fixed<T, length> vec) {
	out << "(";
	int i;
	for (i = 0; i < length - 1; i++) {
		out << vec(i) << ", ";
	}
	out << vec(i) << ")";
	return out;
}

template<typename T>
class array2d {
public:
	array2d(int width, int height) {
		this->width = width;
		this->height = height;
		data = new T[width * height];
	}

	array2d(const array2d<T> &rhs) {
		width = rhs.width;
		height = rhs.height;
		data = new T[width * height];
		for (int j = 0; j < height; j++)
			for (int i = 0; i < width; i++)
				(*this)(i, j) = rhs.data[j * width + i];
	}

	~array2d() {
		delete[] data;
	}

	T &operator()(int col, int row) {
		return data[row * width + col];
	}

	int get_width() { return width; }

	int get_height() { return height; }

	array2d<T> &operator*=(T scalar) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				(*this)(i, j) *= scalar;
			}
		}
		return *this;
	}

	array2d<T> operator*(T scalar) {
		array2d<T> result(*this);
		result *= scalar;
		return result;
	}

	vector<T> operator*(vector<T> vec) {
		vector<T> result;
		T sum;
		for (int row = 0; row < get_height(); row++) {
			sum = 0;
			for (int col = 0; col < get_width(); col++) {
				sum += (*this)(col, row) * vec[col];
			}
			result.push_back(sum);
		}
		return result;
	}

	array2d<T> &multiply_row_scalar(int row, float mult) {
		for (int i = 0; i < get_width(); i++) {
			(*this)(i, row) *= mult;
		}
		return *this;
	}

	array2d<T> &add_row_multiple(int from_row, int to_row, float mult) {
		for (int i = 0; i < get_width(); i++) {
			(*this)(i, to_row) += mult * (*this)(i, from_row);
		}
		return *this;
	}

	// We use simple Gaussian elimination - perf doesn't matter since
	// the matrices will be K x K, where K = number of palette entries.
	array2d<T> matrix_inverse() {
		array2d<T> result(get_width(), get_height());
		array2d<T> &a = *this;

		// Set result to identity matrix
		for (int j = 0; j < get_height(); j++)
			for (int i = 0; i < get_width(); i++)
				result(i, j) = 0;
		for (int i = 0; i < get_width(); i++)
			result(i, i) = 1;

		// Reduce to echelon form, mirroring in result
		for (int i = 0; i < get_width(); i++) {
			result.multiply_row_scalar(i, 1 / a(i, i));
			multiply_row_scalar(i, 1 / a(i, i));
			for (int j = i + 1; j < get_height(); j++) {
				result.add_row_multiple(i, j, -a(i, j));
				add_row_multiple(i, j, -a(i, j));
			}
		}
		// Back substitute, mirroring in result
		for (int i = get_width() - 1; i >= 0; i--) {
			for (int j = i - 1; j >= 0; j--) {
				result.add_row_multiple(i, j, -a(i, j));
				add_row_multiple(i, j, -a(i, j));
			}
		}
		// result is now the inverse
		return result;
	}

private:
	T *data;
	int width, height;
};

template<typename T>
array2d<T> operator*(T scalar, array2d<T> a) {
	return a * scalar;
}


template<typename T>
ostream &operator<<(ostream &out, array2d<T> &a) {
	out << "(";
	int i, j;
	for (j = 0; j < a.get_height(); j++) {
		out << "(";
		for (i = 0; i < a.get_width() - 1; i++) {
			out << a(i, j) << ", ";
		}
		if (j == a.get_height() - 1) {
			out << a(i, j) << "))" << endl;
		} else {
			out << a(i, j) << ")," << endl << " ";
		}
	}
	return out;
}

template<typename T>
class array3d {
public:
	array3d(int width, int height, int depth) {
		this->width = width;
		this->height = height;
		this->depth = depth;
		data = new T[width * height * depth];
	}

	array3d(const array3d<T> &rhs) {
		width = rhs.width;
		height = rhs.height;
		depth = rhs.depth;
		data = new T[width * height * depth];
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				for (int k = 0; k < depth; k++)
					(*this)(i, j, k) = rhs.data[j * width * depth + i * depth + k];
			}
		}
	}

	~array3d() {
		delete[] data;
	}

	T &operator()(int col, int row, int layer) {
		return data[row * width * depth + col * depth + layer];
	}

	int get_width() { return width; }

	int get_height() { return height; }

	int get_depth() { return depth; }

private:
	T *data;
	int width, height, depth;
};

template<typename T>
ostream &operator<<(ostream &out, array3d<T> &a) {
	out << "(";
	int i, j, k;
	out << "(";
	for (j = 0; j <= a.get_height() - 1; j++) {
		out << "(";
		for (i = 0; i <= a.get_width() - 1; i++) {
			out << "(";
			for (k = 0; k <= a.get_depth() - 1; k++) {
				out << a(i, j, k);
				if (k < a.get_depth() - 1) out << ", ";
			}
			out << ")";
			if (i < a.get_height() - 1) out << ",";
		}
		out << ")";
		if (j < a.get_height() - 1) out << ", " << endl;
	}
	out << ")" << endl;
	return out;
}

float b_value(array2d<float> &b, int i_x, int i_y, int j_x, int j_y) {
	int radius_width = (b.get_width() - 1) / 2,
		radius_height = (b.get_height() - 1) / 2;
	int k_x = j_x - i_x + radius_width;
	int k_y = j_y - i_y + radius_height;
	if (k_x >= 0 && k_y >= 0 && k_x < b.get_width() && k_y < b.get_height())
		return b(k_x, k_y);
	else
		return 0.0;
}

template<typename T, int length>
vector<T> extract_vector_layer_1d(vector<vector_fixed<T, length> > s, int k) {
	vector<T> result;
	for (int i = 0; i < s.size(); i++) {
		result.push_back(s[i](k));
	}
	return result;
}

const char *timeAsString(void) {
	static char tstr[64];

	time_t t = time(0);
	struct tm *tm = localtime(&t);
	strftime(tstr, sizeof(tstr), "%Y-%m-%d %H:%M:%S", tm);

	return tstr;
}

const char *arg_inputName;
int arg_paletteSize;
const char *arg_outputName;
const char *opt_paletteName = NULL;
const char *opt_opaqueName = NULL;
const char *opt_snapshotName = NULL;
float opt_stddev = 1.0;
int opt_filterSize = 5;
float opt_initialTemperature = 1;
float opt_finalTemperature = 1e-36;
int opt_numLevels = 30;
int opt_repeatsPerLevel = 6;
int opt_verbose = 1;
int opt_seed = 0;
int opt_visit2 = 0;
int opt_freeze = 0;

void compute_initial_s(array2d<float> &s,
		       array3d<float> &coarse,
		       array2d<float> &weights) {
	int paletteSize = s.get_width();
	int width = coarse.get_width();
	int height = coarse.get_height();
	int filterSize = weights.get_width();

	int center_x = (filterSize - 1) / 2, center_y = (filterSize - 1) / 2;
	float center_b = b_value(weights, 0, 0, 0, 0);

	for (int v = 0; v < paletteSize; v++) {
		for (int alpha = v; alpha < paletteSize; alpha++)
			s(v, alpha) = 0.0;
	}
	for (int i_y = 0; i_y < height; i_y++) {
		for (int i_x = 0; i_x < width; i_x++) {
			int max_j_x = min(width, i_x - center_x + filterSize);
			int max_j_y = min(height, i_y - center_y + filterSize);
			for (int j_y = max(0, i_y - center_y); j_y < max_j_y; j_y++) {
				for (int j_x = max(0, i_x - center_x); j_x < max_j_x; j_x++) {
					if (i_x == j_x && i_y == j_y) continue;
					float b_ij = b_value(weights, i_x, i_y, j_x, j_y);
					for (int v = 0; v < paletteSize; v++) {
						for (int alpha = v; alpha < paletteSize; alpha++)
							s(v, alpha) += b_ij * coarse(i_x, i_y, v) * coarse(j_x, j_y, alpha);
					}
				}
			}
			for (int v = 0; v < paletteSize; v++)
				s(v, v) += center_b * coarse(i_x, i_y, v);
		}
	}
}

void refine_palette(array2d<float> &s,
		    array3d<float> &coarse,
		    array2d<vector_fixed<float, 3> > &image,
		    vector<vector_fixed<float, 3> > &palette) {
	int paletteSize = palette.size();

	// We only computed the half of S above the diagonal - reflect it
	for (int v = 0; v < s.get_width(); v++) {
		for (int alpha = 0; alpha < v; alpha++) {
			s(v, alpha) = s(alpha, v);
		}
	}

	vector<vector_fixed<float, 3> > r(paletteSize);
	for (int v = 0; v < paletteSize; v++) {
		for (int i_y = 0; i_y < coarse.get_height(); i_y++) {
			for (int i_x = 0; i_x < coarse.get_width(); i_x++) {
				r[v] += coarse(i_x, i_y, v) * image(i_x, i_y);
			}
		}
	}

	array2d<float> sInv = s.matrix_inverse();

	for (int k = 0; k < 3; k++) {
		vector<float> R_k = extract_vector_layer_1d(r, k);
		vector<float> palette_channel = sInv * R_k;
		for (int v = opt_freeze; v < paletteSize; v++) {
			float val = palette_channel[v];
			if (val < 0) val = 0;
			if (val > 1) val = 1;

			palette[v](k) = round(val * 255.0) / 255.0;
		}
	}
}

void spatial_color_quant(array2d<vector_fixed<float, 3> > &image,
			 array2d<float> &weights,
			 array2d<int> &quantized_image,
			 vector<vector_fixed<float, 3> > &palette,
			 array3d<float> &coarse,
			 const uint8_t *enabledPixels) {

	const int width = image.get_width();
	const int height = image.get_height();
	const int size2d = width * height;
	const int filterSize = weights.get_width();
	const int paletteSize = palette.size();

	// force change detection by invalidating output image
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			quantized_image(x, y) = paletteSize;
		}
	}

	float temperature = opt_initialTemperature;
	float temperature_multiplier = pow(opt_finalTemperature / opt_initialTemperature, 1.0 / (opt_numLevels - 1));

	char visit_pending[size2d]; // indexed by y*width+x
	int visit_xy[size2d];
	int visit_in = 0;
	int visit_out = 0;
	int visit_cnt = 0;
	int visit_tick = 0;

	array2d<vector_fixed<float, 3> > j_palette_sum(width, height);
	float meanfield_logs[MAXPALETTE];
	float meanfields[MAXPALETTE];

	int pixelsChanged = 0, pixelsVisited = 0;
	for (int iLevel = 0; iLevel < opt_numLevels; iLevel++, temperature *= temperature_multiplier) {
		int levelChanged = 0;

		float middle_b = b_value(weights, 0, 0, 0, 0);

		// construct new palette for this level
		if (iLevel > 0) {
			array2d<float> s(paletteSize, paletteSize);
			compute_initial_s(s, coarse, weights);
			refine_palette(s, coarse, image, palette);
		}

		// compute_initial_j_palette_sum
		for (int j_y = 0; j_y < height; j_y++) {
			for (int j_x = 0; j_x < width; j_x++) {
				vector_fixed<float, 3> palette_sum = vector_fixed<float, 3>();
				for (int alpha = 0; alpha < paletteSize; alpha++)
					palette_sum += coarse(j_x, j_y, alpha) * palette[alpha];
				j_palette_sum(j_x, j_y) = palette_sum;
			}
		}

		int center_x = (filterSize - 1) / 2, center_y = (filterSize - 1) / 2;
		for (int iRepeat = 0; iRepeat < opt_repeatsPerLevel; iRepeat++) {
			int repeatChanged = 0;

			// put enabled pixels in queue
			visit_cnt = 0;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int xy = y * width + x;

					if (enabledPixels[xy >> 3] & (1 << (xy & 7))) {
						visit_pending[y * width + x] = 1;
						visit_xy[visit_cnt++] = xy;
					}
				}
			}
			visit_out = 0;
			visit_in = visit_cnt % size2d; // next position to add to queue
			visit_tick = visit_cnt; // when to generate next log message

			// randomize
			for (int i = 1; i < visit_cnt; i++) {
				int j = rand() % (i + 1);

				int swap = visit_xy[i];
				visit_xy[i] = visit_xy[j];
				visit_xy[j] = swap;
			}

			// Compute 2*sum(j in extended neighborhood of i, j != i) b_ij

			int maxLoop = 100; // in case when stuck in local minimum
			while (visit_cnt) {
				/*
				 * Get next location to visit
				 */
				const int i_x = visit_xy[visit_out] % width;
				const int i_y = visit_xy[visit_out] / width;
				const int xy = i_y * width + i_x;
				visit_out = (visit_out + 1) % size2d;
				visit_cnt--;

				// test if pixel was enabled
				if (!(enabledPixels[xy >> 3] & (1 << (xy & 7))))
					continue;

				// Compute (25)
				vector_fixed<float, 3> p_i;
				for (int y = 0; y < filterSize; y++) {
					for (int x = 0; x < filterSize; x++) {
						int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
						if (i_x == j_x && i_y == j_y) continue;
						if (j_x < 0 || j_y < 0 || j_x >= width || j_y >= height) continue;
						p_i += b_value(weights, i_x, i_y, j_x, j_y) * j_palette_sum(j_x, j_y);
					}
				}
				p_i -= image(i_x, i_y);
				p_i *= 2.0;

				float max_meanfield_log = -numeric_limits<float>::infinity();
				for (int v = 0; v < paletteSize; v++) {
					// Update m_{pi(i)v}^I according to (23)
					// We can subtract an arbitrary factor to prevent overflow,
					// since only the weight relative to the sum matters, so we
					// will choose a value that makes the maximum e^100. (e^80 for `float`)

					float m = -(palette[v].dot_product(p_i + middle_b * palette[v])) / temperature;

					meanfield_logs[v] = m;
					if (m > max_meanfield_log)
						max_meanfield_log = m;
				}

				float meanfield_sum = 0.0;
				for (int v = 0; v < paletteSize; v++) {
					float m = exp(meanfield_logs[v] - max_meanfield_log + 80);

					meanfields[v] = m;
					meanfield_sum += m;
				}
				if (meanfield_sum == 0) {
					cout << "Fatal error: Meanfield sum underflowed. Please contact developer." << endl;
					exit(-1);
				}

				// update variables and determine new palette index for pixel
				int max_v = 0;
				float max_weight = -1;

				vector_fixed<float, 3> &j_pal = j_palette_sum(i_x, i_y);
				for (int v = 0; v < paletteSize; v++) {
					float new_val = meanfields[v] / meanfield_sum;
					// Prevent the matrix S from becoming singular
					if (new_val <= 0) new_val = 1e-10;
					if (new_val >= 1) new_val = 1 - 1e-10;

					float delta_m_iv = new_val - coarse(i_x, i_y, v);
					j_pal += delta_m_iv * palette[v];

					coarse(i_x, i_y, v) = new_val;

					// keep track of new max_weight
					if (new_val > max_weight) {
						max_v = v;
						max_weight = new_val;
					}
				}

				// did color change?
				if (max_v != quantized_image(i_x, i_y)) {
					pixelsChanged++;
					repeatChanged++;
					levelChanged++;

					// update output image
					quantized_image(i_x, i_y) = max_v;

					// We don't add the outer layer of pixels , because
					// there isn't much weight there, and if it does need
					// to be visited, it'll probably be added when we visit
					// neighboring pixels.
					for (int y = min(1, center_y - 1); y < max(filterSize - 1, center_y + 1); y++) {
						for (int x = min(1, center_x - 1); x < max(filterSize - 1, center_x + 1); x++) {
							int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
							if (j_x < 0 || j_y < 0 || j_x >= width || j_y >= height) continue;

							// revisit pixel
							if (!visit_pending[j_y * width + j_x]) {
								visit_pending[j_y * width + j_x] = 1;
								visit_cnt++;
								visit_xy[visit_in] = j_y * width + j_x;
								visit_in = (visit_in + 1) % size2d;
							}
						}
					}
				}

				// mark pixel as processed
				visit_pending[i_y * width + i_x] = 0;
				pixelsVisited++;

				if (--visit_tick <= 0) {
					if (opt_verbose >= 2) {
						fprintf(stderr, "%s level=%d iRepeat=%d temperature=%g visited=%d changed=%d\n", timeAsString(), iLevel, iRepeat, temperature, pixelsVisited, pixelsChanged);
						pixelsVisited = pixelsChanged = 0;
					}
					visit_tick = visit_cnt;
					if (--maxLoop < 0)
						break;

					/*
					 * NOTE: Adding neighbours usually goes in bursts that might introduce visual grid/stripes
					 *       Randomizing the visit queue more will introduce more noise, and possibly better annealing
					 *       only it might require more repeatsPerLevel (slower) and introduce colour fading for 'rare' colours.
					 */
					if (opt_visit2 && visit_cnt) {
						// fast move to head
						if (visit_in < visit_out) {
							for (int i = visit_out; i < size2d; i++)
								visit_xy[visit_in + i - visit_out] = visit_xy[i];
						} else if (visit_in < visit_out) {
							for (int i = visit_out; i < visit_in; i++)
								visit_xy[i - visit_out] = visit_xy[i];
						}
						visit_out = 0;
						visit_in = visit_cnt % size2d;

						// randomize
						for (int i = 1; i < visit_cnt; i++) {
							int j = rand() % (i + 1);

							int swap = visit_xy[i];
							visit_xy[i] = visit_xy[j];
							visit_xy[j] = swap;
						}
					}
				}
			}

			float diff = 0;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					diff += (image(x, y)(0) - palette[quantized_image(x, y)](0)) * (image(x, y)(0) - palette[quantized_image(x, y)](0));
					diff += (image(x, y)(1) - palette[quantized_image(x, y)](1)) * (image(x, y)(1) - palette[quantized_image(x, y)](1));
					diff += (image(x, y)(2) - palette[quantized_image(x, y)](2)) * (image(x, y)(2) - palette[quantized_image(x, y)](2));
				}
			}

			if (opt_verbose >= 2) {
				fprintf(stderr, "%s level=%d iRepeat=%d temperature=%g visited=%d changed=%d diff=%.8g\n", timeAsString(), iLevel, iRepeat, temperature, pixelsVisited, pixelsChanged, diff);
				pixelsVisited = pixelsChanged = 0;
			}
			if (!repeatChanged)
				break;
		}

		if (opt_snapshotName) {
			char *fname;
			asprintf(&fname, opt_snapshotName, iLevel);

			FILE *fil = fopen(fname, "w");
			if (fil == NULL) {
				fprintf(stderr, "Could not open opaque file \"%s\"\n", fname);
				exit(1);
			}


			gdImagePtr im = gdImageCreateTrueColor(width, height);

			for (int i_y = 0; i_y < height; i_y++) {
				for (int i_x = 0; i_x < width; i_x++) {
					int xy = i_y * width + i_x;

					if (!(enabledPixels[xy >> 3] & (1 << (xy & 7)))) {
						// disabled/transparent pixel

						gdImageSetPixel(im, i_x, i_y, gdImageColorAllocateAlpha(im, 127, 128, 129, 0x7f));
					} else {
						// Compute (25) including center pixel
						vector_fixed<float, 3> sum;
						for (int y = 0; y < filterSize; y++) {
							for (int x = 0; x < filterSize; x++) {
								int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
								if (j_x < 0 || j_y < 0 || j_x >= width || j_y >= height) continue;

								sum += b_value(weights, i_x, i_y, j_x, j_y) * j_palette_sum(j_x, j_y);
							}
						}

						int r = round(sum(0) * 255);
						int g = round(sum(1) * 255);
						int b = round(sum(2) * 255);

						gdImageSetPixel(im, i_x, i_y, gdImageColorAllocate(im, r, g, b));
					}
				}
			}

			gdImagePng(im, fil);
			gdImageDestroy(im);
			fclose(fil);
			free(fname);
		}

		if (opt_verbose >= 3) {
			static int colourCount[MAXPALETTE];
			static int lastCount[MAXPALETTE];
			static vector_fixed<float, 3> lastRGB[MAXPALETTE];

			for (int i = 0; i < paletteSize; i++)
				colourCount[i] = 0;

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int i = quantized_image(x, y);
					colourCount[i]++;
				}
			}
			for (int i = 0; i < paletteSize; i++) {
				vector_fixed<float, 3> diff = palette[i] * 255 - lastRGB[i];

				fprintf(stderr, "%3d(%6d): %3d %3d %3d    (%8d) %4d %4d %4d\n",
					i, colourCount[i],
					(int) (palette[i](0) * 255), (int) (palette[i](1) * 255), (int) (palette[i](2) * 255),
					colourCount[i] - lastCount[i],
					(int) diff(0), (int) diff(1), (int) diff(2)
				);
				lastCount[i] = colourCount[i];
				lastRGB[i] += diff;
			}

		}

		float diff = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				diff += (image(x, y)(0) - palette[quantized_image(x, y)](0)) * (image(x, y)(0) - palette[quantized_image(x, y)](0));
				diff += (image(x, y)(1) - palette[quantized_image(x, y)](1)) * (image(x, y)(1) - palette[quantized_image(x, y)](1));
				diff += (image(x, y)(2) - palette[quantized_image(x, y)](2)) * (image(x, y)(2) - palette[quantized_image(x, y)](2));
			}
		}

		if (opt_verbose == 1) {
			fprintf(stderr, "%s level=%d temperature=%g visited=%d changed=%d diff=%.8g\n", timeAsString(), iLevel, temperature, pixelsVisited, pixelsChanged, diff);
			pixelsVisited = pixelsChanged = 0;
		}
		if (!levelChanged)
			break;
	}
}

#include "octree.h"

void usage(const char *argv0, bool verbose) {
	fprintf(stderr, "usage: %s [options] <input> <paletteSize> <outputGIF>\n", argv0);

	if (verbose) {
		fprintf(stderr, "\n");
		fprintf(stderr, "\t-d --stddev=n               std deviation grid [default=%f]\n", opt_stddev);
		fprintf(stderr, "\t-f --filter=n               Filter 1=1x1, 3=3x3, 5=5x5 [default=%d]\n", opt_filterSize);
		fprintf(stderr, "\t   --freeze=n               Freeze/lock the first n colours of the palette [default=%d]\n", opt_freeze);
		fprintf(stderr, "\t-h --help                   This list\n");
		fprintf(stderr, "\t-l --levels=n               Number of levels [default=%d]\n", opt_numLevels);
		fprintf(stderr, "\t-o --opaque=file            Additional opaque oytput [default=%s]\n", opt_opaqueName ? opt_opaqueName : "");
		fprintf(stderr, "\t-p --palette=file           Fixed palette [default=%s]\n", opt_paletteName ? opt_paletteName : "");
		fprintf(stderr, "\t-q --quiet                  Say less\n");
		fprintf(stderr, "\t-r --repeats=n              Number of repeats per level [default=%d]\n", opt_numLevels);
		fprintf(stderr, "\t   --snapshot=file          Internal snapshots filename template [default=%s]\n", opt_snapshotName ? opt_snapshotName : "");
		fprintf(stderr, "\t-v --verbose                Say more\n");
		fprintf(stderr, "\t   --final-temperature=n    Set final temperature [default=%g]\n", opt_finalTemperature);
		fprintf(stderr, "\t   --initial-temperature=n  Set initial temperature [default=%g]\n", opt_initialTemperature);
		fprintf(stderr, "\t   --visit2                 Randomize visit queue more to remove 'stripes'\n");
	}
}

int main(int argc, char *argv[]) {
	for (;;) {
		int option_index = 0;
		enum {
			LO_INITIALTEMP = 1, LO_FINALTEMP, LO_VISIT2, LO_SNAPSHOT, LO_FREEZE,
			LO_HELP = 'h', LO_VERBOSE = 'v', LO_STDDEV = 'd', LO_SEED = 's', LO_FILTER = 'f', LO_LEVELS = 'l', LO_REPEATS = 'r', LO_PALETTE = 'p', LO_OPAQUE = 'o', LO_QUIET = 'q'
		};
		static struct option long_options[] = {
			/* name, has_arg, flag, val */
			{"filter",              1, 0, LO_FILTER},
			{"final-temperature",   1, 0, LO_FINALTEMP},
			{"freeze",              1, 0, LO_FREEZE},
			{"help",                0, 0, LO_HELP},
			{"initial-temperature", 1, 0, LO_INITIALTEMP},
			{"levels",              1, 0, LO_LEVELS},
			{"opaque",              1, 0, LO_OPAQUE},
			{"palette",             1, 0, LO_PALETTE},
			{"quiet",               0, 0, LO_QUIET},
			{"repeats",             1, 0, LO_REPEATS},
			{"seed",                1, 0, LO_SEED},
			{"snapshot",            1, 0, LO_SNAPSHOT},
			{"stddev",              1, 0, LO_STDDEV},
			{"verbose",             0, 0, LO_VERBOSE},
			{"visit2",              0, 0, LO_VISIT2},
			{NULL,                  0, 0, 0}
		};

		char optstring[128], *cp;
		cp = optstring;
		for (int i = 0; long_options[i].name; i++) {
			if (isalpha(long_options[i].val)) {
				*cp++ = long_options[i].val;
				if (long_options[i].has_arg)
					*cp++ = ':';
			}
		}
		*cp++ = '\0';

		int c = getopt_long(argc, argv, optstring, long_options, &option_index);
		if (c == -1)
			break;

		switch (c) {
			case LO_FILTER:
				opt_filterSize = strtol(optarg, NULL, 10);
				if (opt_filterSize != 1 && opt_filterSize != 3 && opt_filterSize != 5) {
					fprintf(stderr, "Filter size must be one of 1, 3, or 5.\n");
					exit(1);
				}
				break;
			case LO_FINALTEMP:
				opt_finalTemperature = strtod(optarg, NULL);
				break;
			case LO_FREEZE:
				opt_freeze = strtol(optarg, NULL, 10);
				break;
			case LO_HELP:
				usage(argv[0], 0);
				exit(0);
				break;
			case LO_INITIALTEMP:
				opt_initialTemperature = strtod(optarg, NULL);
				break;
			case LO_LEVELS:
				opt_numLevels = strtol(optarg, NULL, 10);
				break;
			case LO_OPAQUE:
				opt_opaqueName = optarg;
				break;
			case LO_PALETTE:
				opt_paletteName = optarg;
				break;
			case LO_QUIET:
				opt_verbose--;
				break;
			case LO_REPEATS:
				opt_repeatsPerLevel = strtol(optarg, NULL, 10);
				break;
			case LO_SEED:
				opt_seed = strtol(optarg, NULL, 10);
				break;
			case LO_SNAPSHOT:
				opt_snapshotName = optarg;
				break;
			case LO_STDDEV:
				opt_stddev = strtod(optarg, NULL);
				if (opt_stddev <= 0) {
					fprintf(stderr, "Std deviation must be > 0\\n");
					exit(1);
				}
				break;
			case LO_VERBOSE:
				opt_verbose++;
				break;
			case LO_VISIT2:
				opt_visit2++;
				break;
			case '?':
				fprintf(stderr, "Try `%s --help' for more information.\n", argv[0]);
				exit(1);
				break;
			default:
				fprintf(stderr, "getopt returned character code %d\n", c);
				exit(1);
		}
	}

	if (argc - optind < 3) {
		usage(argv[0], 1);
		exit(1);
	}

	arg_inputName = argv[optind++];
	arg_paletteSize = strtol(argv[optind++], NULL, 10);
	arg_outputName = argv[optind++];

	if (arg_paletteSize < 2 || arg_paletteSize > MAXPALETTE)
		printf("Number of colors must be at least 2 and no more than %d.\n", MAXPALETTE);

	/*
	 * Set random number generator
	 */
	if (!opt_seed)
		opt_seed = time(NULL);
	srand(opt_seed);

	/*
	 * Construct weights
	 */

	// allocate structures
	array2d<float> filter1_weights(1, 1);
	array2d<float> filter3_weights(3, 3);
	array2d<float> filter5_weights(5, 5);
	array2d<float> *filters[] = {NULL, &filter1_weights, NULL, &filter3_weights, NULL, &filter5_weights};

	filter1_weights(0, 0) = 1.0;

	float sum = 0.0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			sum += filter3_weights(i, j) = exp(-sqrt((i - 1) * (i - 1) + (j - 1) * (j - 1)) / (opt_stddev * opt_stddev));
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			filter3_weights(i, j) /= sum;
		}
	}
	if (opt_verbose) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++)
				printf("%5.2f ", filter3_weights(i, j) / filter3_weights(0, 0));
			printf("\n");
		}
		printf("\n");
	}

	sum = 0.0;
	int coef5[] = {1, 4, 6, 4, 1};
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++)
			sum += filter5_weights(i, j) = exp(-sqrt((i - 2) * (i - 2) + (j - 2) * (j - 2)) / (opt_stddev * opt_stddev));
	}
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++)
			filter5_weights(i, j) /= sum;
	}

	if (opt_verbose) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++)
				printf("%5.2f ", filter5_weights(i, j) / filter5_weights(0, 0));
			printf("\n");
		}
		printf("\n");
	}

	/*
	 * Load input image and extract width/height
	 */

	// open source
	FILE *fil = fopen(arg_inputName, "r");
	if (fil == NULL) {
		fprintf(stderr, "Could not open source image \"%s\"\n", arg_inputName);
		exit(1);
	}

	// probe file type and load
	gdImagePtr im = NULL;
	unsigned char c[2];
	if (fread(c, 2, 1, fil) == 1) {
		rewind(fil);
		if (c[0] == 0x89 && c[1] == 0x50)
			im = gdImageCreateFromPng(fil);
		if (c[0] == 0x47 && c[1] == 0x49)
			im = gdImageCreateFromGif(fil);
		if (c[0] == 0xff && c[1] == 0xd8)
			im = gdImageCreateFromJpeg(fil);
	}
	if (im == NULL) {
		fprintf(stderr, "Could not load source image %x %x\n", c[0], c[1]);
		exit(1);
	}
	fclose(fil);

	// extract image size
	int width = gdImageSX(im);
	int height = gdImageSY(im);
	assert(width > 0 && height > 0);

	// allocate structures
	array2d<vector_fixed<float, 3> > image(width, height);
	array2d<int> quantized_image(width, height);
	vector<vector_fixed<float, 3> > palette;
	uint8_t enabledPixels[((width * height) >> 3) + 1];
	memset(enabledPixels, 0, ((width * height) >> 3) + 1);

	int transparent = -1;

	/*
	 * load image
	 */
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int xy = y * width + x;
			int v = gdImageGetTrueColorPixel(im, x, y);

			int a = (v >> 24) & 0xff; // 0=opaque 0x7f=transparent
			int r = (v >> 16) & 0xff;
			int g = (v >> 8) & 0xff;
			int b = (v >> 0) & 0xff;

			if (a) {
				// load gray50 into a0
				image(x, y)(0) = 0.5;
				image(x, y)(1) = 0.5;
				image(x, y)(2) = 0.5;

				if (transparent < 0) {
					// use first free colour for
					transparent = arg_paletteSize;
				}

			} else {
				// load gray50 into a0
				image(x, y)(0) = r / 255.0;
				image(x, y)(1) = g / 255.0;
				image(x, y)(2) = b / 255.0;

				// add endpoint to nodeHeap
				heapAdd(&nodeHeap, nodeInsert(r, g, b));

				// enable pixel
				enabledPixels[xy >> 3] |= 1 << (xy & 7);
			}
		}
	}
	gdImageDestroy(im);

	/*
	 * construct initial palette
	 */
	if (!opt_paletteName) {
		/*
		 * Octree palette
		 */

		// merge colours until final palette count (heap contains empty root node)
		while (nodeHeap.count - 1 > arg_paletteSize)
			nodeFold(heapPop(&nodeHeap));

		// inject colours into palette
		for (int i = 1; i < nodeHeap.count; i++) {
			octNode_t *got = nodeHeap.ppNodes[i];

			got->r = got->r / got->count;
			got->g = got->g / got->count;
			got->b = got->b / got->count;

			printf("%3ld | %3u %3u %3u (%d pixels)\n", i - 1, got->r, got->g, got->b, got->count);

			vector_fixed<float, 3> v;
			v(0) = got->r / 255.0;
			v(1) = got->g / 255.0;
			v(2) = got->b / 255.0;
			palette.push_back(v);
		}

	} else if (strcmp(opt_paletteName, "random") == 0) {
		/*
		 * Random palette
		 */

		for (int i = 0; i < arg_paletteSize; i++) {
			vector_fixed<float, 3> v;
			v(0) = ((float) rand()) / RAND_MAX;
			v(1) = ((float) rand()) / RAND_MAX;
			v(2) = ((float) rand()) / RAND_MAX;
			palette.push_back(v);
		}

	} else {
		/*
		 * Load palette from file
		 */
		FILE *in = fopen(opt_paletteName, "r");
		if (in == NULL) {
			fprintf(stderr, "Could not open palette \"%s\"\n", opt_paletteName);
			exit(1);
		}

		arg_paletteSize = 0;

		float r, g, b;
		char line[256];
		int lineNr = 0;
		while (fgets(line, sizeof line, in)) {
			lineNr++;
			if (sscanf(line, "%f %f %f\n", &r, &g, &b) == 3) {
				if (arg_paletteSize >= MAXPALETTE) {
					fprintf(stderr, "too many colours in palette \"%s\"\n", opt_paletteName);
					exit(1);
				}

				if (r > 1 || g > 1 || b > 1) {
					// integer 0-255
					r /= 255.0;
					g /= 255.0;
					b /= 255.0;
				} else {
					// float 0-1
				}

				vector_fixed<float, 3> v;
				v(0) = r;
				v(1) = g;
				v(2) = b;
				palette.push_back(v);

				arg_paletteSize++;
			} else {
				fprintf(stderr, "Error in line %d of palette \"%s\"\n", lineNr, opt_paletteName);
				exit(1);
			}
		}

		fclose(in);
	}

	char *pJson;
	asprintf(&pJson, "{\"srcName\":\"%s\",\"width\":%d,\"height\":%d,\"paletteSize\":%d,\"transparent\":%d,\"seed\":%d,\"filter\":%d,\"numLevels\":%d,\"repeatsPerLevel\":%d,\"initialTemperature\":%g,\"finalTemperature\":%g,\"stddef\":%f,\"palette\":\"%s\",\"freeze\":%d,\"visit2\":%d}\n",
		 arg_inputName,
		 width, height,
		 arg_paletteSize, transparent,
		 opt_seed, opt_filterSize,
		 opt_numLevels, opt_repeatsPerLevel, opt_initialTemperature, opt_finalTemperature,
		 opt_stddev,
		 opt_paletteName ? opt_paletteName : "",
		 opt_freeze,
		 opt_visit2
	);
	fprintf(stderr, "%s", pJson);

	assert(palette.size() == arg_paletteSize);

	array3d<float> coarse(width, height, arg_paletteSize);

	// with octree, no need to randomize coarse
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < arg_paletteSize; k++)
				coarse(i, j, k) = 1.0 / arg_paletteSize;
		}
	}

	spatial_color_quant(image, *filters[opt_filterSize], quantized_image, palette, coarse, enabledPixels);

	// frequency count colours
	static int colourCount[MAXPALETTE];
	for (int i = 0; i < arg_paletteSize; i++) colourCount[i] = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int i = quantized_image(x, y);
			colourCount[i]++;
		}
	}

	if (arg_outputName) {
		/*
		 * Save image
		 */
		FILE *fil = fopen(arg_outputName, "w");
		if (fil == NULL) {
			fprintf(stderr, "Could not open output file \"%s\"\n", arg_outputName);
			exit(1);
		}

		gdImagePtr im = gdImageCreate(width, height);

		// pre-allocate palette
		for (int i = 0; i < arg_paletteSize; i++) {
			int ix = gdImageColorAllocate(im, palette[i](0) * 255, palette[i](1) * 255, palette[i](2) * 255);
			assert(ix == i);
		}
		if (transparent >= 0) {
			// for transparency, add extra colour
			int ix = gdImageColorAllocateAlpha(im, 126, 127, 128, 0x7f); // 0=opaque 0x7f=transparent
			assert(ix == transparent);
			assert(ix == arg_paletteSize);
			gdImageColorTransparent(im, transparent);

			palette[arg_paletteSize](0) = 126 / 255.0;
			palette[arg_paletteSize](1) = 127 / 255.0;
			palette[arg_paletteSize](2) = 128 / 255.0;
			arg_paletteSize++;
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int xy = y * width + x;

				if (!(enabledPixels[xy >> 3] & (1 << (xy & 7)))) {
					// disabled/transparent pixel
					gdImageSetPixel(im, x, y, transparent);
				} else {
					// enabled pixel
					int c = quantized_image(x, y);
					assert(c >= 0 && c <= arg_paletteSize);

					gdImageSetPixel(im, x, y, c);
				}
			}
		}

		gdImageGif(im, fil);
		gdImageDestroy(im);
		fclose(fil);

		/*
		 * Save palette
		 */
		char *fname;
		asprintf(&fname, "%s.pal", arg_outputName);
		fil = fopen(fname, "w");
		if (fil == NULL) {
			fprintf(stderr, "Could not open output file \"%s\"\n", fname);
			exit(1);
		}

		for (int i = 0; i < arg_paletteSize; i++)
			fprintf(fil, "%3d %3d %3d # %d\n", (int) (palette[i](0) * 255), (int) (palette[i](1) * 255), (int) (palette[i](2) * 255), colourCount[i]);

		fclose(fil);
		free(fname);

		/*
		 * Save json
		 */
		asprintf(&fname, "%s.json", arg_outputName);
		fil = fopen(fname, "w");
		if (fil == NULL) {
			fprintf(stderr, "Could not open output file \"%s\"\n", fname);
			exit(1);
		}
		fprintf(fil, "%s", pJson);

		fclose(fil);
		free(fname);
	}

	if (opt_opaqueName) {
		FILE *fil = fopen(opt_opaqueName, "w");
		if (fil == NULL) {
			fprintf(stderr, "Could not open opaque file \"%s\"\n", opt_opaqueName);
			exit(1);
		}

		gdImagePtr im = gdImageCreateTrueColor(width, height);

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int xy = y * width + x;

				if (!(enabledPixels[xy >> 3] & (1 << (xy & 7)))) {
					// disabled/transparent pixel

					gdImageSetPixel(im, x, y, gdImageColorAllocateAlpha(im, 127, 128, 129, 0x7f));
				} else {
					// enabled pixel
					int c = quantized_image(x, y);
					int r, g, b;

					r = round(palette[c](0) * 255);
					g = round(palette[c](1) * 255);
					b = round(palette[c](2) * 255);

					gdImageSetPixel(im, x, y, gdImageColorAllocate(im, r, g, b));
				}
			}
		}

		gdImagePng(im, fil);
		gdImageDestroy(im);
		fclose(fil);
	}

	if (opt_verbose) {
		for (int i = 0; i < arg_paletteSize; i++)
			fprintf(stderr, "%3d(%6d): %3d %3d %3d\n", i, colourCount[i], (int) (palette[i](0) * 255), (int) (palette[i](1) * 255), (int) (palette[i](2) * 255));
	}

	return 0;
}
