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

	int get_length() { return length; }

	T norm_squared() {
		T result = 0;
		for (int i = 0; i < length; i++) {
			result += (*this)(i) * (*this)(i);
		}
		return result;
	}

	vector_fixed<T, length> &operator=(const vector_fixed<T, length> rhs) {
		for (int i = 0; i < length; i++) {
			data[i] = rhs.data[i];
		}
		return *this;
	}

	vector_fixed<T, length> direct_product(vector_fixed<T, length> &rhs) {
		vector_fixed<T, length> result;
		for (int i = 0; i < length; i++) {
			result(i) = (*this)(i) * rhs(i);
		}
		return result;
	}

	double dot_product(vector_fixed<T, length> rhs) {
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
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				(*this)(i, j) = rhs.data[j * width + i];
			}
		}
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
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
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

	array2d<T> &multiply_row_scalar(int row, double mult) {
		for (int i = 0; i < get_width(); i++) {
			(*this)(i, row) *= mult;
		}
		return *this;
	}

	array2d<T> &add_row_multiple(int from_row, int to_row, double mult) {
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
		result *= 0;
		for (int i = 0; i < get_width(); i++) {
			result(i, i) = 1;
		}
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
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < depth; k++) {
					(*this)(i, j, k) = rhs.data[j * width * depth + i * depth + k];
				}
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

int compute_max_coarse_level(int width, int height) {
	// We want the coarsest layer to have at most MAX_PIXELS pixels
	const int MAX_PIXELS = 4000;
	int result = 0;
	while (width * height > MAX_PIXELS) {
		width >>= 1;
		height >>= 1;
		result++;
	}
	return result;
}

void fill_random(array3d<double> &a) {
	for (int i = 0; i < a.get_width(); i++) {
		for (int j = 0; j < a.get_height(); j++) {
			for (int k = 0; k < a.get_depth(); k++) {
				a(i, j, k) = ((double) rand()) / RAND_MAX;
			}
		}
	}
}

void random_permutation(int count, vector<int> &result) {
	result.clear();
	for (int i = 0; i < count; i++) {
		result.push_back(i);
	}
	random_shuffle(result.begin(), result.end());
}

void random_permutation_2d(int width, int height, deque<pair<int, int> > &result) {
	vector<int> perm1d;
	random_permutation(width * height, perm1d);
	while (!perm1d.empty()) {
		int idx = perm1d.back();
		perm1d.pop_back();
		result.push_back(pair<int, int>(idx % width, idx / width));
	}
}

void compute_b_array(array2d<vector_fixed<double, 3> > &filter_weights,
		     array2d<vector_fixed<double, 3> > &b) {
	// Assume that the pixel i is always located at the center of b,
	// and vary pixel j's location through each location in b.
	int radius_width = (filter_weights.get_width() - 1) / 2,
		radius_height = (filter_weights.get_height() - 1) / 2;
	int offset_x = (b.get_width() - 1) / 2 - radius_width;
	int offset_y = (b.get_height() - 1) / 2 - radius_height;
	for (int j_y = 0; j_y < b.get_height(); j_y++) {
		for (int j_x = 0; j_x < b.get_width(); j_x++) {
			for (int k_y = 0; k_y < filter_weights.get_height(); k_y++) {
				for (int k_x = 0; k_x < filter_weights.get_width(); k_x++) {
					if (k_x + offset_x >= j_x - radius_width &&
					    k_x + offset_x <= j_x + radius_width &&
					    k_y + offset_y >= j_y - radius_width &&
					    k_y + offset_y <= j_y + radius_width) {
						b(j_x, j_y) += filter_weights(k_x, k_y).direct_product(filter_weights(k_x + offset_x - j_x + radius_width, k_y + offset_y - j_y + radius_height));
					}
				}
			}
		}
	}
}

vector_fixed<double, 3> b_value(array2d<vector_fixed<double, 3> > &b,
				int i_x, int i_y, int j_x, int j_y) {
	int radius_width = (b.get_width() - 1) / 2,
		radius_height = (b.get_height() - 1) / 2;
	int k_x = j_x - i_x + radius_width;
	int k_y = j_y - i_y + radius_height;
	if (k_x >= 0 && k_y >= 0 && k_x < b.get_width() && k_y < b.get_height())
		return b(k_x, k_y);
	else
		return vector_fixed<double, 3>();
}

void compute_a_image(array2d<vector_fixed<double, 3> > &image,
		     array2d<vector_fixed<double, 3> > &b,
		     array2d<vector_fixed<double, 3> > &a) {
	int radius_width = (b.get_width() - 1) / 2,
		radius_height = (b.get_height() - 1) / 2;
	for (int i_y = 0; i_y < a.get_height(); i_y++) {
		for (int i_x = 0; i_x < a.get_width(); i_x++) {
			for (int j_y = i_y - radius_height; j_y <= i_y + radius_height; j_y++) {
				if (j_y < 0) j_y = 0;
				if (j_y >= a.get_height()) break;

				for (int j_x = i_x - radius_width; j_x <= i_x + radius_width; j_x++) {
					if (j_x < 0) j_x = 0;
					if (j_x >= a.get_width()) break;

					a(i_x, i_y) += b_value(b, i_x, i_y, j_x, j_y).direct_product(image(j_x, j_y));
				}
			}
			a(i_x, i_y) *= -2.0;
		}
	}
}

void sum_coarsen(array2d<vector_fixed<double, 3> > &fine,
		 array2d<vector_fixed<double, 3> > &coarse) {
	for (int y = 0; y < coarse.get_height(); y++) {
		for (int x = 0; x < coarse.get_width(); x++) {
			double divisor = 1.0;
			vector_fixed<double, 3> val = fine(x * 2, y * 2);
			if (x * 2 + 1 < fine.get_width()) {
				divisor += 1;
				val += fine(x * 2 + 1, y * 2);
			}
			if (y * 2 + 1 < fine.get_height()) {
				divisor += 1;
				val += fine(x * 2, y * 2 + 1);
			}
			if (x * 2 + 1 < fine.get_width() &&
			    y * 2 + 1 < fine.get_height()) {
				divisor += 1;
				val += fine(x * 2 + 1, y * 2 + 1);
			}
			coarse(x, y) = /*(1/divisor)**/val;
		}
	}
}

template<typename T, int length>
array2d<T> extract_vector_layer_2d(array2d<vector_fixed<T, length> > s, int k) {
	array2d<T> result(s.get_width(), s.get_height());
	for (int i = 0; i < s.get_width(); i++) {
		for (int j = 0; j < s.get_height(); j++) {
			result(i, j) = s(i, j)(k);
		}
	}
	return result;
}

template<typename T, int length>
vector<T> extract_vector_layer_1d(vector<vector_fixed<T, length> > s, int k) {
	vector<T> result;
	for (unsigned int i = 0; i < s.size(); i++) {
		result.push_back(s[i](k));
	}
	return result;
}

int best_match_color(array3d<double> &vars, int i_x, int i_y,
		     vector<vector_fixed<double, 3> > &palette) {
	int max_v = 0;
	double max_weight = vars(i_x, i_y, 0);
	for (unsigned int v = 1; v < palette.size(); v++) {
		if (vars(i_x, i_y, v) > max_weight) {
			max_v = v;
			max_weight = vars(i_x, i_y, v);
		}
	}
	return max_v;
}

void zoom_double(array3d<double> &small, array3d<double> &big) {
	// Simple scaling of the weights array based on mixing the four
	// pixels falling under each fine pixel, weighted by area.
	// To mix the pixels a little, we assume each fine pixel
	// is 1.2 fine pixels wide and high.
	for (int y = 0; y < big.get_height() / 2 * 2; y++) {
		for (int x = 0; x < big.get_width() / 2 * 2; x++) {
			double left = max(0.0, (x - 0.1) / 2.0), right = min(small.get_width() - 0.001, (x + 1.1) / 2.0);
			double top = max(0.0, (y - 0.1) / 2.0), bottom = min(small.get_height() - 0.001, (y + 1.1) / 2.0);
			int x_left = (int) floor(left), x_right = (int) floor(right);
			int y_top = (int) floor(top), y_bottom = (int) floor(bottom);
			double area = (right - left) * (bottom - top);
			double top_left_weight = (ceil(left) - left) * (ceil(top) - top) / area;
			double top_right_weight = (right - floor(right)) * (ceil(top) - top) / area;
			double bottom_left_weight = (ceil(left) - left) * (bottom - floor(bottom)) / area;
			double bottom_right_weight = (right - floor(right)) * (bottom - floor(bottom)) / area;
			double top_weight = (right - left) * (ceil(top) - top) / area;
			double bottom_weight = (right - left) * (bottom - floor(bottom)) / area;
			double left_weight = (bottom - top) * (ceil(left) - left) / area;
			double right_weight = (bottom - top) * (right - floor(right)) / area;
			for (int z = 0; z < big.get_depth(); z++) {
				if (x_left == x_right && y_top == y_bottom) {
					big(x, y, z) = small(x_left, y_top, z);
				} else if (x_left == x_right) {
					big(x, y, z) = top_weight * small(x_left, y_top, z) +
						       bottom_weight * small(x_left, y_bottom, z);
				} else if (y_top == y_bottom) {
					big(x, y, z) = left_weight * small(x_left, y_top, z) +
						       right_weight * small(x_right, y_top, z);
				} else {
					big(x, y, z) = top_left_weight * small(x_left, y_top, z) +
						       top_right_weight * small(x_right, y_top, z) +
						       bottom_left_weight * small(x_left, y_bottom, z) +
						       bottom_right_weight * small(x_right, y_bottom, z);
				}
			}
		}
	}
}

void compute_initial_s(array2d<vector_fixed<double, 3> > &s,
		       array3d<double> &coarse_variables,
		       array2d<vector_fixed<double, 3> > &b) {
	int palette_size = s.get_width();
	int coarse_width = coarse_variables.get_width();
	int coarse_height = coarse_variables.get_height();
	int center_x = (b.get_width() - 1) / 2, center_y = (b.get_height() - 1) / 2;
	vector_fixed<double, 3> center_b = b_value(b, 0, 0, 0, 0);
	vector_fixed<double, 3> zero_vector;
	for (int v = 0; v < palette_size; v++) {
		for (int alpha = v; alpha < palette_size; alpha++) {
			s(v, alpha) = zero_vector;
		}
	}
	for (int i_y = 0; i_y < coarse_height; i_y++) {
		for (int i_x = 0; i_x < coarse_width; i_x++) {
			int max_j_x = min(coarse_width, i_x - center_x + b.get_width());
			int max_j_y = min(coarse_height, i_y - center_y + b.get_height());
			for (int j_y = max(0, i_y - center_y); j_y < max_j_y; j_y++) {
				for (int j_x = max(0, i_x - center_x); j_x < max_j_x; j_x++) {
					if (i_x == j_x && i_y == j_y) continue;
					vector_fixed<double, 3> b_ij = b_value(b, i_x, i_y, j_x, j_y);
					for (int v = 0; v < palette_size; v++) {
						for (int alpha = v; alpha < palette_size; alpha++) {
							double mult = coarse_variables(i_x, i_y, v) * coarse_variables(j_x, j_y, alpha);
							s(v, alpha)(0) += mult * b_ij(0);
							s(v, alpha)(1) += mult * b_ij(1);
							s(v, alpha)(2) += mult * b_ij(2);
						}
					}
				}
			}
			for (int v = 0; v < palette_size; v++) {
				s(v, v) += coarse_variables(i_x, i_y, v) * center_b;
			}
		}
	}
}

void update_s(array2d<vector_fixed<double, 3> > &s,
	      array3d<double> &coarse_variables,
	      array2d<vector_fixed<double, 3> > &b,
	      int j_x, int j_y, int alpha,
	      double delta) {
	int palette_size = s.get_width();
	int coarse_width = coarse_variables.get_width();
	int coarse_height = coarse_variables.get_height();
	int center_x = (b.get_width() - 1) / 2, center_y = (b.get_height() - 1) / 2;
	int max_i_x = min(coarse_width, j_x + center_x + 1);
	int max_i_y = min(coarse_height, j_y + center_y + 1);
	for (int i_y = max(0, j_y - center_y); i_y < max_i_y; i_y++) {
		for (int i_x = max(0, j_x - center_x); i_x < max_i_x; i_x++) {
			vector_fixed<double, 3> delta_b_ij = delta * b_value(b, i_x, i_y, j_x, j_y);
			if (i_x == j_x && i_y == j_y) continue;
			for (int v = 0; v <= alpha; v++) {
				double mult = coarse_variables(i_x, i_y, v);
				s(v, alpha)(0) += mult * delta_b_ij(0);
				s(v, alpha)(1) += mult * delta_b_ij(1);
				s(v, alpha)(2) += mult * delta_b_ij(2);
			}
			for (int v = alpha; v < palette_size; v++) {
				double mult = coarse_variables(i_x, i_y, v);
				s(alpha, v)(0) += mult * delta_b_ij(0);
				s(alpha, v)(1) += mult * delta_b_ij(1);
				s(alpha, v)(2) += mult * delta_b_ij(2);
			}
		}
	}
	s(alpha, alpha) += delta * b_value(b, 0, 0, 0, 0);
}

void refine_palette(array2d<vector_fixed<double, 3> > &s,
		    array3d<double> &coarse_variables,
		    array2d<vector_fixed<double, 3> > &a,
		    vector<vector_fixed<double, 3> > &palette) {
	// We only computed the half of S above the diagonal - reflect it
	for (int v = 0; v < s.get_width(); v++) {
		for (int alpha = 0; alpha < v; alpha++) {
			s(v, alpha) = s(alpha, v);
		}
	}

	vector<vector_fixed<double, 3> > r(palette.size());
	for (unsigned int v = 0; v < palette.size(); v++) {
		for (int i_y = 0; i_y < coarse_variables.get_height(); i_y++) {
			for (int i_x = 0; i_x < coarse_variables.get_width(); i_x++) {
				r[v] += coarse_variables(i_x, i_y, v) * a(i_x, i_y);
			}
		}
	}

	for (unsigned int k = 0; k < 3; k++) {
		array2d<double> S_k = extract_vector_layer_2d(s, k);
		vector<double> R_k = extract_vector_layer_1d(r, k);
		vector<double> palette_channel = -1.0 * ((2.0 * S_k).matrix_inverse()) * R_k;
		for (unsigned int v = 0; v < palette.size(); v++) {
			double val = palette_channel[v];
			if (val < 0) val = 0;
			if (val > 1) val = 1;
			palette[v](k) = val;
		}
	}

#if TRACE
	for (unsigned int v=0; v<palette.size(); v++) {
	    cout << palette[v] << endl;
	}
#endif
}

void compute_initial_j_palette_sum(array2d<vector_fixed<double, 3> > &j_palette_sum,
				   array3d<double> &coarse_variables,
				   vector<vector_fixed<double, 3> > &palette) {
	for (int j_y = 0; j_y < coarse_variables.get_height(); j_y++) {
		for (int j_x = 0; j_x < coarse_variables.get_width(); j_x++) {
			vector_fixed<double, 3> palette_sum = vector_fixed<double, 3>();
			for (unsigned int alpha = 0; alpha < palette.size(); alpha++) {
				palette_sum += coarse_variables(j_x, j_y, alpha) * palette[alpha];
			}
			j_palette_sum(j_x, j_y) = palette_sum;
		}
	}
}

void spatial_color_quant(array2d<vector_fixed<double, 3> > &image,
			 array2d<vector_fixed<double, 3> > &filter_weights,
			 array2d<int> &quantized_image,
			 vector<vector_fixed<double, 3> > &palette,
			 array3d<double> *&p_coarse_variables,
			 double initial_temperature,
			 double final_temperature,
			 int temps_per_level,
			 int repeats_per_temp,
			 int verbose) {
	int max_coarse_level = //1;
		compute_max_coarse_level(image.get_width(), image.get_height());
	p_coarse_variables = new array3d<double>(
		image.get_width() >> max_coarse_level,
		image.get_height() >> max_coarse_level,
		palette.size());
	// For syntactic convenience
	array3d<double> &coarse_variables = *p_coarse_variables;
	fill_random(coarse_variables);

	double temperature = initial_temperature;

	// Compute a_i, b_{ij} according to (11)
	int extended_neighborhood_width = filter_weights.get_width() * 2 - 1;
	int extended_neighborhood_height = filter_weights.get_height() * 2 - 1;
	array2d<vector_fixed<double, 3> > b0(extended_neighborhood_width, extended_neighborhood_height);
	compute_b_array(filter_weights, b0);

	array2d<vector_fixed<double, 3> > a0(image.get_width(), image.get_height());
	compute_a_image(image, b0, a0);

	// Compute a_I^l, b_{IJ}^l according to (18)
	vector<array2d<vector_fixed<double, 3> > > a_vec, b_vec;
	a_vec.push_back(a0);
	b_vec.push_back(b0);

	int coarse_level;
	for (coarse_level = 1; coarse_level <= max_coarse_level; coarse_level++) {
		int radius_width = (filter_weights.get_width() - 1) / 2,
			radius_height = (filter_weights.get_height() - 1) / 2;
		array2d<vector_fixed<double, 3> > bi(max(3, b_vec.back().get_width() - 2), max(3, b_vec.back().get_height() - 2));
		for (int J_y = 0; J_y < bi.get_height(); J_y++) {
			for (int J_x = 0; J_x < bi.get_width(); J_x++) {
				for (int i_y = radius_height * 2; i_y < radius_height * 2 + 2; i_y++) {
					for (int i_x = radius_width * 2; i_x < radius_width * 2 + 2; i_x++) {
						for (int j_y = J_y * 2; j_y < J_y * 2 + 2; j_y++) {
							for (int j_x = J_x * 2; j_x < J_x * 2 + 2; j_x++) {
								bi(J_x, J_y) += b_value(b_vec.back(), i_x, i_y, j_x, j_y);
							}
						}
					}
				}
			}
		}
		b_vec.push_back(bi);

		array2d<vector_fixed<double, 3> > ai(image.get_width() >> coarse_level, image.get_height() >> coarse_level);
		sum_coarsen(a_vec.back(), ai);
		a_vec.push_back(ai);
	}

	// Multiscale annealing
	coarse_level = max_coarse_level;
	const int iters_per_level = temps_per_level;
	double temperature_multiplier = pow(final_temperature / initial_temperature, 1.0 / (max(3, max_coarse_level * iters_per_level)));

	if (verbose)
		cout << "Temperature multiplier: " << temperature_multiplier << endl;

	int iters_at_current_level = 0;
	bool skip_palette_maintenance = false;
	array2d<vector_fixed<double, 3> > s(palette.size(), palette.size());
	compute_initial_s(s, coarse_variables, b_vec[coarse_level]);
	array2d<vector_fixed<double, 3> > *j_palette_sum = new array2d<vector_fixed<double, 3> >(coarse_variables.get_width(), coarse_variables.get_height());
	compute_initial_j_palette_sum(*j_palette_sum, coarse_variables, palette);
	while (coarse_level >= 0 || temperature > final_temperature) {
		// Need to reseat this reference in case we changed p_coarse_variables
		array3d<double> &coarse_variables = *p_coarse_variables;
		array2d<vector_fixed<double, 3> > &a = a_vec[coarse_level];
		array2d<vector_fixed<double, 3> > &b = b_vec[coarse_level];
		vector_fixed<double, 3> middle_b = b_value(b, 0, 0, 0, 0);

		if (verbose)
			cout << "Temperature: " << temperature << endl;

		int center_x = (b.get_width() - 1) / 2, center_y = (b.get_height() - 1) / 2;
		int step_counter = 0;
		for (int repeat = 0; repeat < repeats_per_temp; repeat++) {
			int pixels_changed = 0, pixels_visited = 0;
			deque<pair<int, int> > visit_queue;
			random_permutation_2d(coarse_variables.get_width(), coarse_variables.get_height(), visit_queue);

			// Compute 2*sum(j in extended neighborhood of i, j != i) b_ij

			while (!visit_queue.empty()) {
				// If we get to 10% above initial size, just revisit them all
				if ((int) visit_queue.size() > coarse_variables.get_width() * coarse_variables.get_height() * 11 / 10) {
					visit_queue.clear();
					random_permutation_2d(coarse_variables.get_width(), coarse_variables.get_height(), visit_queue);
				}

				int i_x = visit_queue.front().first, i_y = visit_queue.front().second;
				visit_queue.pop_front();

				// Compute (25)
				vector_fixed<double, 3> p_i;
				for (int y = 0; y < b.get_height(); y++) {
					for (int x = 0; x < b.get_width(); x++) {
						int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
						if (i_x == j_x && i_y == j_y) continue;
						if (j_x < 0 || j_y < 0 || j_x >= coarse_variables.get_width() || j_y >= coarse_variables.get_height()) continue;
						vector_fixed<double, 3> b_ij = b_value(b, i_x, i_y, j_x, j_y);
						vector_fixed<double, 3> j_pal = (*j_palette_sum)(j_x, j_y);
						p_i(0) += b_ij(0) * j_pal(0);
						p_i(1) += b_ij(1) * j_pal(1);
						p_i(2) += b_ij(2) * j_pal(2);
					}
				}
				p_i *= 2.0;
				p_i += a(i_x, i_y);

				vector<double> meanfield_logs, meanfields;
				double max_meanfield_log = -numeric_limits<double>::infinity();
				double meanfield_sum = 0.0;
				for (unsigned int v = 0; v < palette.size(); v++) {
					// Update m_{pi(i)v}^I according to (23)
					// We can subtract an arbitrary factor to prevent overflow,
					// since only the weight relative to the sum matters, so we
					// will choose a value that makes the maximum e^100.
					meanfield_logs.push_back(-(palette[v].dot_product(p_i + middle_b.direct_product(palette[v]))) / temperature);
					if (meanfield_logs.back() > max_meanfield_log)
						max_meanfield_log = meanfield_logs.back();
				}
				for (unsigned int v = 0; v < palette.size(); v++) {
					meanfields.push_back(exp(meanfield_logs[v] - max_meanfield_log + 100));
					meanfield_sum += meanfields.back();
				}
				if (meanfield_sum == 0) {
					cout << "Fatal error: Meanfield sum underflowed. Please contact developer." << endl;
					exit(-1);
				}
				int old_max_v = best_match_color(coarse_variables, i_x, i_y, palette);
				vector_fixed<double, 3> &j_pal = (*j_palette_sum)(i_x, i_y);
				for (unsigned int v = 0; v < palette.size(); v++) {
					double new_val = meanfields[v] / meanfield_sum;
					// Prevent the matrix S from becoming singular
					if (new_val <= 0) new_val = 1e-10;
					if (new_val >= 1) new_val = 1 - 1e-10;
					double delta_m_iv = new_val - coarse_variables(i_x, i_y, v);
					coarse_variables(i_x, i_y, v) = new_val;
					j_pal(0) += delta_m_iv * palette[v](0);
					j_pal(1) += delta_m_iv * palette[v](1);
					j_pal(2) += delta_m_iv * palette[v](2);
					if (fabs(delta_m_iv) > 0.001 && !skip_palette_maintenance)
						update_s(s, coarse_variables, b, i_x, i_y, v, delta_m_iv);
				}
				int max_v = best_match_color(coarse_variables, i_x, i_y, palette);
				// Only consider it a change if the colors are different enough
				if ((palette[max_v] - palette[old_max_v]).norm_squared() >= 1.0 / (255.0 * 255.0)) {
					pixels_changed++;
					// We don't add the outer layer of pixels , because
					// there isn't much weight there, and if it does need
					// to be visited, it'll probably be added when we visit
					// neighboring pixels.
					// The commented out loops are faster but cause a little bit of distortion
					//for (int y=center_y-1; y<center_y+1; y++) {
					//   for (int x=center_x-1; x<center_x+1; x++) {
					for (int y = min(1, center_y - 1); y < max(b.get_height() - 1, center_y + 1); y++) {
						for (int x = min(1, center_x - 1); x < max(b.get_width() - 1, center_x + 1); x++) {
							int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
							if (j_x < 0 || j_y < 0 || j_x >= coarse_variables.get_width() || j_y >= coarse_variables.get_height()) continue;
							visit_queue.push_back(pair<int, int>(j_x, j_y));
						}
					}
				}
				pixels_visited++;

				// Show progress with dots - in a graphical interface,
				// we'd show progressive refinements of the image instead,
				// and maybe a palette preview.
				step_counter++;
				if ((step_counter % 10000) == 0) {
					cout << ".";
					cout.flush();
#if TRACE
					cout << visit_queue.size();
#endif
				}
			}

			if (verbose > 1)
				cout << "Pixels changed: " << pixels_changed << endl;

			if (skip_palette_maintenance)
				compute_initial_s(s, *p_coarse_variables, b_vec[coarse_level]);
			refine_palette(s, coarse_variables, a, palette);
			compute_initial_j_palette_sum(*j_palette_sum, coarse_variables, palette);
		}

		iters_at_current_level++;
		skip_palette_maintenance = false;
		if ((temperature <= final_temperature || coarse_level > 0) && iters_at_current_level >= iters_per_level) {
			coarse_level--;
			if (coarse_level < 0) break;
			array3d<double> *p_new_coarse_variables = new array3d<double>(
				image.get_width() >> coarse_level,
				image.get_height() >> coarse_level,
				palette.size());
			zoom_double(coarse_variables, *p_new_coarse_variables);
			delete p_coarse_variables;
			p_coarse_variables = p_new_coarse_variables;
			iters_at_current_level = 0;
			delete j_palette_sum;
			j_palette_sum = new array2d<vector_fixed<double, 3> >((*p_coarse_variables).get_width(), (*p_coarse_variables).get_height());
			compute_initial_j_palette_sum(*j_palette_sum, *p_coarse_variables, palette);
			skip_palette_maintenance = true;
#ifdef TRACE
			cout << "Image size: " << p_coarse_variables->get_width() << " " << p_coarse_variables->get_height() << endl;
#endif
		}
		if (temperature > final_temperature)
			temperature *= temperature_multiplier;
	}

	// This is normally not used, but is handy sometimes for debugging
	while (coarse_level > 0) {
		coarse_level--;
		array3d<double> *p_new_coarse_variables = new array3d<double>(
			image.get_width() >> coarse_level,
			image.get_height() >> coarse_level,
			palette.size());
		zoom_double(*p_coarse_variables, *p_new_coarse_variables);
		delete p_coarse_variables;
		p_coarse_variables = p_new_coarse_variables;
	}

	{
		// Need to reseat this reference in case we changed p_coarse_variables
		array3d<double> &coarse_variables = *p_coarse_variables;

		for (int i_x = 0; i_x < image.get_width(); i_x++) {
			for (int i_y = 0; i_y < image.get_height(); i_y++) {
				quantized_image(i_x, i_y) = best_match_color(coarse_variables, i_x, i_y, palette);
			}
		}
		for (unsigned int v = 0; v < palette.size(); v++) {
			for (unsigned int k = 0; k < 3; k++) {
				if (palette[v](k) > 1.0) palette[v](k) = 1.0;
				if (palette[v](k) < 0.0) palette[v](k) = 0.0;
			}
#ifdef TRACE
			cout << palette[v] << endl;
#endif
		}
	}
}

#include "octree.h"

const char *arg_inputName;
int arg_paletteSize;
const char *arg_outputName;
const char *opt_palette = NULL;
const char *opt_opaque = NULL;
float opt_stddev = 1;
int opt_filterSize = 3;
double opt_initialTemperature = 1.0;
double opt_finalTemperature = .001;
int opt_numLevels = 3;
int opt_repeatsPerLevel = 1;
int opt_verbose = 1;
int opt_seed = 0;

void usage(const char *argv0, bool verbose) {
	fprintf(stderr, "usage: %s [options] <input> <paletteSize> <outputGIF>\n", argv0);

	if (verbose) {
		fprintf(stderr, "\n");
		fprintf(stderr, "\t-d --stddev=n               std deviation grid [default=%f]\n", opt_stddev);
		fprintf(stderr, "\t-f --filter=n               Filter 1=1x1, 3=3x3, 5=5x5 [default=%d]\n", opt_filterSize);
		fprintf(stderr, "\t-h --help                   This list\n");
		fprintf(stderr, "\t-l --levels=n               Number of levels [default=%d]\n", opt_numLevels);
		fprintf(stderr, "\t-o --opaque=file            Additional opaque oytput [default=%s]\n", opt_opaque ? opt_opaque : "");
		fprintf(stderr, "\t-p --palette=file           Fixed palette [default=%s]\n", opt_palette ? opt_palette : "");
		fprintf(stderr, "\t-q --quiet                  Say less\n");
		fprintf(stderr, "\t-r --repeats=n              Number of repeats per level [default=%d]\n", opt_numLevels);
		fprintf(stderr, "\t-v --verbose                Say more\n");
		fprintf(stderr, "\t   --final-temperature=n    Set final temperature [default=%f]\n", opt_finalTemperature);
		fprintf(stderr, "\t   --initial-temperature=n  Set initial temperature [default=%f]\n", opt_initialTemperature);
	}
}

int main(int argc, char *argv[]) {
	for (;;) {
		int option_index = 0;
		enum {
			LO_INITIALTEMP = 1, LO_FINALTEMP,
			LO_HELP = 'h', LO_VERBOSE = 'v', LO_STDDEV = 'd', LO_SEED = 's', LO_FILTER = 'f', LO_LEVELS = 'l', LO_REPEATS = 'r', LO_PALETTE = 'p', LO_OPAQUE = 'o', LO_QUIET = 'q'
		};
		static struct option long_options[] = {
			/* name, has_arg, flag, val */
			{"filter",              1, 0, LO_FILTER},
			{"final-temperature",   1, 0, LO_FINALTEMP},
			{"help",                0, 0, LO_HELP},
			{"initial-temperature", 1, 0, LO_INITIALTEMP},
			{"levels",              1, 0, LO_LEVELS},
			{"opaque",              1, 0, LO_OPAQUE},
			{"palette",             1, 0, LO_PALETTE},
			{"quiet",               0, 0, LO_QUIET},
			{"repeats",             1, 0, LO_REPEATS},
			{"seed",                1, 0, LO_SEED},
			{"stddev",              1, 0, LO_STDDEV},
			{"verbose",             0, 0, LO_VERBOSE},
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
			case LO_STDDEV:
				opt_stddev = strtod(optarg, NULL);
				if (opt_stddev <= 0) {
					fprintf(stderr, "Std deviation must be > 0\\n");
					exit(1);
				}
				break;
			case LO_INITIALTEMP:
				opt_initialTemperature = strtod(optarg, NULL);
				break;
			case LO_FINALTEMP:
				opt_finalTemperature = strtod(optarg, NULL);
				break;
			case LO_LEVELS:
				opt_numLevels = strtol(optarg, NULL, 10);
				break;
			case LO_REPEATS:
				opt_repeatsPerLevel = strtol(optarg, NULL, 10);
				break;
			case LO_PALETTE:
				opt_palette = optarg;
				break;
			case LO_OPAQUE:
				opt_opaque = optarg;
				break;
			case LO_SEED:
				opt_seed = strtol(optarg, NULL, 10);
				break;
			case LO_FILTER:
				opt_filterSize = strtol(optarg, NULL, 10);
				if (opt_filterSize != 1 && opt_filterSize != 3 && opt_filterSize != 5) {
					fprintf(stderr, "Filter size must be one of 1, 3, or 5.\n");
					exit(1);
				}
				break;

			case LO_VERBOSE:
				opt_verbose++;
				break;
			case LO_QUIET:
				opt_verbose--;
				break;
			case LO_HELP:
				usage(argv[0], 0);
				exit(0);
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
	array2d<vector_fixed<double, 3> > filter1_weights(1, 1);
	array2d<vector_fixed<double, 3> > filter3_weights(3, 3);
	array2d<vector_fixed<double, 3> > filter5_weights(5, 5);
	array2d<vector_fixed<double, 3> > *filters[] = {NULL, &filter1_weights, NULL, &filter3_weights, NULL, &filter5_weights};

	filter1_weights(0, 0)(0) = 1.0;
	filter1_weights(0, 0)(1) = 1.0;
	filter1_weights(0, 0)(2) = 1.0;

	double sum = 0.0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			double w = exp(-sqrt((i - 1) * (i - 1) + (j - 1) * (j - 1)) / (opt_stddev * opt_stddev));
			sum += w;
			filter3_weights(i, j)(0) = w;
			filter3_weights(i, j)(1) = w;
			filter3_weights(i, j)(2) = w;
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			filter3_weights(i, j)(0) /= sum;
			filter3_weights(i, j)(1) /= sum;
			filter3_weights(i, j)(2) /= sum;
		}
	}
	if (opt_verbose) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				printf("%5.2f ", filter3_weights(i, j)(0) / filter3_weights(0, 0)(0));
			}
			printf("\n");
		}
		printf("\n");
	}

	sum = 0.0;
	int coef5[] = {1, 4, 6, 4, 1};
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			double w = exp(-sqrt((i - 2) * (i - 2) + (j - 2) * (j - 2)) / (opt_stddev * opt_stddev));
			sum += w;
			filter5_weights(i, j)(0) = w;
			filter5_weights(i, j)(1) = w;
			filter5_weights(i, j)(2) = w;
		}
	}
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			filter5_weights(i, j)(0) /= sum;
			filter5_weights(i, j)(1) /= sum;
			filter5_weights(i, j)(2) /= sum;
		}
	}

	if (opt_verbose) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%5.2f ", filter5_weights(i, j)(0) / filter5_weights(0, 0)(0));
			}
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
	array2d<vector_fixed<double, 3> > image(width, height);
	array2d<int> quantized_image(width, height);
	vector<vector_fixed<double, 3> > palette;
	uint8_t enabledPixels[((width * height) >> 3) + 1];
	memset(enabledPixels, 0, ((width * height) >> 3) + 1);

	int numColours = arg_paletteSize;
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
					// remove a colour for use as transparency
					transparent = --numColours;
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
	if (!opt_palette) {
		/*
		 * Octree palette
		 */

		// merge colours until final palette count (heap contains empty root node)
		while (nodeHeap.count - 1 > numColours)
			nodeFold(heapPop(&nodeHeap));

		// inject colours into palette
		for (int i = 1; i < nodeHeap.count; i++) {
			octNode_t *got = nodeHeap.ppNodes[i];

			got->r = got->r / got->count;
			got->g = got->g / got->count;
			got->b = got->b / got->count;

			printf("%3ld | %3u %3u %3u (%d pixels)\n", i - 1, got->r, got->g, got->b, got->count);

			vector_fixed<double, 3> v;
			v(0) = got->r / 255.0;
			v(1) = got->g / 255.0;
			v(2) = got->b / 255.0;
			palette.push_back(v);
		}

	} else if (strcmp(opt_palette, "random") == 0) {
		/*
		 * Random palette
		 */

		for (int i = 0; i < numColours; i++) {
			vector_fixed<double, 3> v;
			v(0) = ((double) rand()) / RAND_MAX;
			v(1) = ((double) rand()) / RAND_MAX;
			v(2) = ((double) rand()) / RAND_MAX;
			palette.push_back(v);
		}

	} else {
		/*
		 * Load palette from file
		 */
		FILE *in = fopen(opt_palette, "r");
		if (in == NULL) {
			fprintf(stderr, "Could not open palette \"%s\"\n", opt_palette);
			exit(1);
		}

		numColours = 0;

		float r, g, b;
		while (fscanf(in, "%f %f %f\n", &r, &g, &b) == 3) {
			if (arg_paletteSize >= MAXPALETTE) {
				fprintf(stderr, "too many colours in palette \"%s\"\n", opt_palette);
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

			vector_fixed<double, 3> v;
			v(0) = r;
			v(1) = g;
			v(2) = b;
			palette.push_back(v);

			numColours++;
		}

		arg_paletteSize = numColours;
		if (transparent >= 0) {
			transparent = arg_paletteSize++;

			fclose(in);
		}
	}

	// for transparency, add extra colour
	if (transparent >= 0) {
		vector_fixed<double, 3> v;
		v(0) = 126 / 255.0;
		v(1) = 127 / 255.0;
		v(2) = 128 / 255.0;
		palette.push_back(v);
	}
	assert(palette.size() == arg_paletteSize);

	array3d<double> *coarse;

	fprintf(stderr, "{srcName:\"%s\",width:%d,height:%d,paletteSize:%d,transparent:%d,seed:%d,filter:%d,numLevels:%d,repeatsPerLevel:%d,initialTemperature:%g,finalTemperature:%g,stddef=%f,palette=\"%s\"}\n",
		arg_inputName,
		width, height,
		arg_paletteSize, transparent,
		opt_seed, opt_filterSize,
		opt_numLevels, opt_repeatsPerLevel, opt_initialTemperature, opt_finalTemperature,
		opt_stddev,
		opt_palette ? opt_palette : ""
	);

	spatial_color_quant(image, *filters[opt_filterSize], quantized_image, palette, coarse, opt_initialTemperature,
			    opt_finalTemperature, opt_numLevels, opt_repeatsPerLevel, opt_verbose);

	if (arg_outputName) {
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
		if (transparent >= 0)
			gdImageColorTransparent(im, transparent);

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int xy = y * width + x;

				if (!(enabledPixels[xy >> 3] & (1 << (xy & 7)))) {
					// disabled/transparent pixel
					gdImageSetPixel(im, x, y, transparent);
				} else {
					// enabled pixel
					int c = quantized_image(x, y);

					gdImageSetPixel(im, x, y, c);
				}
			}
		}

		gdImageGif(im, fil);
		gdImageDestroy(im);
		fclose(fil);
	}

	if (opt_opaque) {
		FILE *fil = fopen(opt_opaque, "w");
		if (fil == NULL) {
			fprintf(stderr, "Could not open opaque file \"%s\"\n", opt_opaque);
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

	if (1) {
		static int colourCount[256];

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int i = quantized_image(x, y);
				colourCount[i]++;
			}
		}
		for (int i = 0; i < arg_paletteSize; i++)
			fprintf(stderr, "%3d(%6d): %3d %3d %3d\n", i, colourCount[i], (int) (palette[i](0) * 255), (int) (palette[i](1) * 255), (int) (palette[i](2) * 255));
	}

	return 0;
}
