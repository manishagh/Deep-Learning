// HW5.cpp : Defines the entry point for the console application.
//
#define _CRT_SECURE_NO_DEPRECATE
//#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <math.h>
#include <algorithm> 
using namespace std;

double sigmoid(double x)
{
	double exp_value;
	double return_value;

	/*** Exponential calculation ***/
	exp_value = exp((double)-x);

	/*** Final tanh value ***/
	return_value = 1 / (1 + exp_value);

	return return_value;
}
double relu(double x)
{
	if (x > 0)
		return x;
	else
		return 0.01*x;
}
double act(double x)
{
	return tanh(x);
}

double der_relu(double x)
{
	/*if (x > 0)
		return 1;
	else
		return 0.01;*/
	return(1 - x*x);
}
double der_act(double x)
{
	return der_relu(x);
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(int nimage, int idata, std::vector<vector<double>> &arr, std::vector<vector<int>> &lbl)
{


	arr.resize(nimage, vector<double>(idata));
	std::ifstream infile("train-images.idx3-ubyte",ios::binary); 

	if (infile.is_open())
	{
		
		int magic = 0;
		int imageNo = 0;
		int width = 0;
		int height = 0;
		infile.read((char*)&magic, sizeof(magic));
		magic = ReverseInt(magic);
		infile.read((char*)&imageNo, sizeof(imageNo));
		imageNo = ReverseInt(imageNo);
		infile.read((char*)&width, sizeof(width));
		width = ReverseInt(width);
		infile.read((char*)&height, sizeof(height));
		height = ReverseInt(height);
		for (int i = 0; i < imageNo; ++i)
		{
			for (int r = 0; r < width; ++r)
			{
				for (int c = 0; c < height; ++c)
				{
					unsigned char temp = 0;
					infile.read((char*)&temp, sizeof(temp));
					if (i == 0)
						cout << int(temp) << " ";
					arr[i][(width*r) + c] = (double)((temp)*1.0)/255;
					
				}
			}
		}
	}
	infile.close();
	lbl.resize(60000, vector<int>(10));
	std::ifstream lblfile("train-labels.idx1-ubyte", ios::binary);
	lbl.resize(60000, vector<int>(10));
	if (lblfile.is_open())
	{
		
		int magic = 0;
		int imageNo = 0;
		int width = 0;
		int height = 0;
		lblfile.read((char*)&magic, sizeof(magic));
		magic = ReverseInt(magic);
		lblfile.read((char*)&imageNo, sizeof(imageNo));
		imageNo = ReverseInt(imageNo);
		for (int i = 0; i < imageNo; ++i)
		{
			unsigned char temp = 0;
			lblfile.read((char*)&temp, sizeof(temp));
			int j;
			for (int r = 0; r < 10; ++r)
			{
				if (r == int(temp))
					lbl[i][r] = 1;
				else
					lbl[i][r] = 0;	
			}
		}
	}
	lblfile.close();
}

void ReadMNISTlabel(int nimage, int idata, std::vector<vector<double>> &arr, std::vector<vector<int>> &lbl)
{


	arr.resize(nimage, vector<double>(idata));
	std::ifstream infile("t10k-images.idx3-ubyte", ios::binary);

	if (infile.is_open())
	{
		
		int magic = 0;
		int imageNo = 0;
		int width = 0;
		int height = 0;
		infile.read((char*)&magic, sizeof(magic));
		magic = ReverseInt(magic);
		infile.read((char*)&imageNo, sizeof(imageNo));
		imageNo = ReverseInt(imageNo);
		infile.read((char*)&width, sizeof(width));
		width = ReverseInt(width);
		infile.read((char*)&height, sizeof(height));
		height = ReverseInt(height);
		for (int i = 0; i < imageNo; ++i)
		{
			for (int r = 0; r < width; ++r)
			{
				for (int c = 0; c < height; ++c)
				{
					unsigned char temp = 0;
					infile.read((char*)&temp, sizeof(temp));
					arr[i][(width*r) + c] = double(temp) / 255;
				}
			}
		}
	}
	infile.close();
	lbl.resize(60000, vector<int>(10));
	std::ifstream lblfile("t10k-labels.idx1-ubyte", ios::binary);
	if (lblfile.is_open())
	{
		int magic = 0;
		int imageNo = 0;
		int width = 0;
		int height = 0;
		lblfile.read((char*)&magic, sizeof(magic));
		magic = ReverseInt(magic);
		lblfile.read((char*)&imageNo, sizeof(imageNo));
		imageNo = ReverseInt(imageNo);
		for (int i = 0; i < imageNo; ++i)
		{
			unsigned char temp = 0;
			lblfile.read((char*)&temp, sizeof(temp));
			int j;
			for (int r = 0; r < 10; ++r)
			{
				if (r == int(temp))
					lbl[i][r] = 1;
				else
					lbl[i][r] = 0;
			}
		}
	}
	lblfile.close();
}
int main()
{
	std::vector<std::vector<double>> dataarr1;
	std::vector<std::vector<int>> labels;
	std::vector<std::vector<double>> testarr1;
	std::vector<std::vector<int>> testlabel;
	double dataarr[28][28];
	double testarr[28][28];
	printf("reading data\n");
	
	ReadMNIST(60000, 784, dataarr1, labels);
	
	
	ReadMNISTlabel(10000, 784, testarr1, testlabel);
	
	ofstream outfile("file2.dat");

	
	//int max = INT_MAX;
	double w0[5][5], w1[5][5], w2[5][5], w3[5][5], w4[5][5], w5[5][5],w6[5][5],w7[5][5];
	double w00[5][5], w10[5][5], w20[5][5], w30[5][5], w40[5][5], w50[5][5],w60[5][5],w70[5][5];
	double  b00, b01, b02, b03, b04, b05,b06,b07;
	double  b000, b010, b020, b030, b040,b050,b060,b070;
	double y00[24][24], y01[24][24], y02[24][24], y03[24][24], y04[24][24], y05[24][24], y06[24][24], y07[24][24];
	double y00s[24][24], y01s[24][24],y02s[24][24],y03s[24][24],y04s[24][24],y05s[24][24],y06s[24][24], y07s[24][24];
	double max0[12][12],max1[12][12],max2[12][12],max3[12][12],max4[12][12],max5[12][12], max6[12][12], max7[12][12];
	double u0[144][44], u1[144][44], u2[144][44], u3[144][44], u4[144][44], u5[144][44], u6[144][44], u7[144][44];
	double u00[144][44],u10[144][44],u20[144][44],u30[144][44],u40[144][44], u50[144][44], u60[144][44], u70[144][44];
	double b10[44];
	double b100[44];
	double y10[44];
	double y10s[44];
	double v0[44][10];
	double v00[44][10];
	double b20[10];
	double b200[10];
	double y20[10];

	std::vector<double> z;
	z.resize(10);

	int maxindex0[12][12],maxindex1[12][12],maxindex2[12][12],maxindex3[12][12],maxindex4[12][12],maxindex5[12][12],
		maxindex6[12][12], maxindex7[12][12];

	

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			w0[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w0[" << i << "][" << j << "]=" << w0[i][j] << endl;
			w1[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w1[" << i << "][" << j << "]=" << w1[i][j] << endl;
			w2[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w2[" << i << "][" << j << "]=" << w2[i][j] << endl;
			w3[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w3[" << i << "][" << j << "]=" << w3[i][j] << endl;
			w4[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w4[" << i << "][" << j << "]=" << w4[i][j] << endl;
			w5[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w5[" << i << "][" << j << "]=" << w5[i][j] << endl;
			w6[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w6[" << i << "][" << j << "]=" << w6[i][j] << endl;
			w7[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "w7[" << i << "][" << j << "]=" << w7[i][j] << endl;
			
			
		}
	}
	b00 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile <<"b00="<< b00 << endl;
	b01 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b01=" << b01 << endl;
	b02 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b02=" << b02 << endl;
	b03 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b03=" << b03 << endl;
	b04 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b04=" << b04 << endl;
	b05 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b05=" << b05 << endl;
	b06 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b06=" << b06 << endl;
	b07 = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
	outfile << "b07=" << b07 << endl;
	

	for (int i = 0; i < 144; i++)
	{
		for (int j = 0; j < 44; j++)
		{
			u0[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u0[" << i << "][" << j << "]=" << u0[i][j] << endl;
			u1[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u1[" << i << "][" << j << "]=" << u1[i][j] << endl;
			u2[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u2[" << i << "][" << j << "]=" << u2[i][j] << endl;
			u3[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u3[" << i << "][" << j << "]=" << u3[i][j] << endl;
			u4[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u4[" << i << "][" << j << "]=" << u4[i][j] << endl;
			u5[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u5[" << i << "][" << j << "]=" << u5[i][j] << endl;
			u6[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u6[" << i << "][" << j << "]=" << u6[i][j] << endl;
			u7[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "u7[" << i << "][" << j << "]=" << u7[i][j] << endl;
		}
	}
	
	for (int i = 0; i < 44; i++)
	{
		b10[i] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
		outfile << "b10[" << i << "]=" << b10[i] << endl;

	}

	
	for (int i = 0; i < 44; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			v0[i][j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
			outfile << "v0[" << i << "][" << j << "]=" << v0[i][j] << endl;

		}
	}
	
	for (int j = 0; j < 10; j++)
	{
		b20[j] = (double(rand()) / double(RAND_MAX)) * 2 * 0.12 - 0.12;
		outfile << "b20[" << j << "]=" << b20[j] << endl;
	}

	//learning rate
	double	e = 0.01;
	outfile << "learning rate= " << e << endl;
	for (int d3 = 0; d3 < 40; d3++)
	{
		outfile << "BEGINING OF EPOCH " << d3 + 1 << endl;
		printf("begin epoch %d\n", d3 + 1);
		//training
		for (int r = 0; r < 60000; r++)
		{
			for (int k = 0; k < 28; k++)
				for (int l = 0; l < 28; l++)
					dataarr[k][l] = dataarr1[r][l + k * 28];

			//Convolution layer calculation--//forward propagation

			for (int i = 0; i < 24; i++)
			{
				for (int j = 0; j < 24; j++)
				{
					y00[i][j] = 0;
					y01[i][j] = 0;
					y02[i][j] = 0;
					y03[i][j] = 0;
					y04[i][j] = 0;
					y05[i][j] = 0;
					y06[i][j] = 0;
					y07[i][j] = 0;
					
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							int p = i + k;
							int s = j + l;
							y00[i][j] = y00[i][j] + w0[k][l] * dataarr[p][s];
							y01[i][j] = y01[i][j] + w1[k][l] * dataarr[p][s];
							y02[i][j] = y02[i][j] + w2[k][l] * dataarr[p][s];
							y03[i][j] = y03[i][j] + w3[k][l] * dataarr[p][s];
							y04[i][j] = y04[i][j] + w4[k][l] * dataarr[p][s];
							y05[i][j] = y05[i][j] + w5[k][l] * dataarr[p][s];
							y06[i][j] = y06[i][j] + w6[k][l] * dataarr[p][s];
							y07[i][j] = y07[i][j] + w7[k][l] * dataarr[p][s];

						}

					}
					y00[i][j] = y00[i][j] + b00;
					y01[i][j] = y01[i][j] + b01;
					y02[i][j] = y02[i][j] + b02;
					y03[i][j] = y03[i][j] + b03;
					y04[i][j] = y04[i][j] + b04;
					y05[i][j] = y05[i][j] + b05;
					y06[i][j] = y06[i][j] + b06;
					y07[i][j] = y07[i][j] + b07;

					y00s[i][j] = act(y00[i][j]);
					y01s[i][j] = act(y01[i][j]);
					y02s[i][j] = act(y02[i][j]);
					y03s[i][j] = act(y03[i][j]);
					y04s[i][j] = act(y04[i][j]);
					y05s[i][j] = act(y05[i][j]);
					y06s[i][j] = act(y06[i][j]);
					y07s[i][j] = act(y07[i][j]);


				}
			}


			//MAX POOLING
			for (int i = 0; i < 12; i++)
			{
				for (int j = 0; j < 12; j++)
				{
					max0[i][j] = y00s[2 * i][2 * j];
					max1[i][j] = y01s[2 * i][2 * j];
					max2[i][j] = y02s[2 * i][2 * j];
					max3[i][j] = y03s[2 * i][2 * j];
					max4[i][j] = y04s[2 * i][2 * j];
					max5[i][j] = y05s[2 * i][2 * j];
					max6[i][j] = y06s[2 * i][2 * j];
					max7[i][j] = y07s[2 * i][2 * j];

					maxindex0[i][j] = 0;
					maxindex1[i][j] = 0;
					maxindex2[i][j] = 0;
					maxindex3[i][j] = 0;
					maxindex4[i][j] = 0;
					maxindex5[i][j] = 0;
					maxindex6[i][j] = 0;
					maxindex7[i][j] = 0;


					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							int p = 2 * i + k;
							int s = 2 * j + l;
							if (max0[i][j] < y00s[p][s])
							{
								max0[i][j] = y00s[p][s];
								maxindex0[i][j] = l + k * 2;
							}
							if (max1[i][j] < y01s[p][s])
							{
								max1[i][j] = y01s[p][s];
								maxindex1[i][j] = l + k * 2;
							}
							if (max2[i][j] < y02s[p][s])
							{
								max2[i][j] = y02s[p][s];
								maxindex2[i][j] = l + k * 2;
							}
							if (max3[i][j] < y03s[p][s])
							{
								max3[i][j] = y03s[p][s];
								maxindex3[i][j] = l + k * 2;
							}
							if (max4[i][j] < y04s[p][s])
							{
								max4[i][j] = y04s[p][s];
								maxindex4[i][j] = l + k * 2;
							}
							if (max5[i][j] < y05s[p][s])
							{
								max5[i][j] = y05s[p][s];
								maxindex5[i][j] = l + k * 2;
							}
							if (max6[i][j] < y06s[p][s])
							{
								max6[i][j] = y06s[p][s];
								maxindex6[i][j] = l + k * 2;
							}
							if (max7[i][j] < y07s[p][s])
							{
								max7[i][j] = y07s[p][s];
								maxindex7[i][j] = l + k * 2;
							}

						}

					}


				}
			}
			

			double sigy10[44];
			//Fully connected layer
			for (int i = 0; i < 44; i++)
			{
				y10[i] = 0;
				for (int k = 0; k < 144; k++)
				{
					int p = k / 12;
					int q = k % 12;
					y10[i] = y10[i] + max0[p][q] * u0[k][i]
						+ max1[p][q] * u1[k][i]
						+ max2[p][q] * u2[k][i]
						+ max3[p][q] * u3[k][i]
						+ max4[p][q] * u4[k][i]
						+ max5[p][q] * u5[k][i]+ max6[p][q] * u6[k][i]+ max7[p][q] * u7[k][i];

				}
				y10[i] = y10[i] + b10[i];
				y10s[i] = act(y10[i]);
				sigy10[i] = der_act(y10s[i]);




			}

			double err_lo[10];

			//OUTPUT LAYER
			for (int i = 0; i < 10; i++)
			{
				y20[i] = 0;
				for (int k = 0; k < 44; k++)
				{
					y20[i] = y20[i] + y10s[k] * v0[k][i];
				}
				y20[i] = y20[i] + b20[i];
				z[i] = act(y20[i]);

				err_lo[i] = (z[i]-labels[r][i])*der_act(z[i]);

			}
			

			//BACKWARD PROPAGATION
			for (int i = 0; i < 10; i++)
			{

				b200[i] = b20[i] - e*err_lo[i];

			}
			for (int i = 0; i < 44; i++)
			{
				for (int j = 0; j < 10; j++)
				{
					v00[i][j] = v0[i][j] - e*err_lo[j] * y10s[i];


				}
			}
			double b0_up[44];
			for (int i = 0; i < 44; i++)
			{
				double kk = 0;
				for (int j = 0; j < 10; j++)
				{
					kk = kk + err_lo[j] * v0[i][j];

				}

				b0_up[i] = kk*sigy10[i];
				b100[i] = b10[i] - e*b0_up[i];


			}




			for (int i = 0; i < 144; i++)
			{

				int p = i / 12;
				int q = i % 12;
				for (int k = 0; k < 44; k++)
				{

					u00[i][k] = u0[i][k] - e*b0_up[k] * max0[p][q];
					u10[i][k] = u1[i][k] - e*b0_up[k] * max1[p][q];
					u20[i][k] = u2[i][k] - e*b0_up[k] * max2[p][q];
					u30[i][k] = u3[i][k] - e*b0_up[k] * max3[p][q];
					u40[i][k] = u4[i][k] - e*b0_up[k] * max4[p][q];
					u50[i][k] = u5[i][k] - e*b0_up[k] * max5[p][q];
					u60[i][k] = u6[i][k] - e*b0_up[k] * max6[p][q];
					u70[i][k] = u7[i][k] - e*b0_up[k] * max7[p][q];

				}

			}

			double p00 = 0, p10 = 0, p20 = 0, p30 = 0, p40 = 0, p50 = 0, p60 = 0, p70 = 0;
			double w_up0[144], w_up1[144], w_up2[144], w_up3[144], w_up4[144], w_up5[144], w_up6[144], w_up7[144];
			for (int n = 0; n < 144; n++)
			{
				int t = n / 12;
				int s = n % 12;
				double p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0,p6=0,p7=0;
				for (int l = 0; l < 44; l++)
				{
					p0 = p0 + b0_up[l] * u0[n][l];
					p1 = p1 + b0_up[l] * u1[n][l];
					p2 = p2 + b0_up[l] * u2[n][l];
					p3 = p3 + b0_up[l] * u3[n][l];
					p4 = p4 + b0_up[l] * u4[n][l];
					p5 = p5 + b0_up[l] * u5[n][l];
					p6 = p6 + b0_up[l] * u6[n][l];
					p7 = p7 + b0_up[l] * u7[n][l];

				}
				w_up0[n] = p0*der_act(max0[t][s]) ;
				w_up1[n] = p1*der_act(max1[t][s]);
				w_up2[n] = p2* der_act(max2[t][s]);
				w_up3[n] = p3*der_act(max3[t][s]);
				w_up4[n] = p4*der_act(max4[t][s]);
				w_up5[n] = p5*der_act(max5[t][s]);
				w_up6[n] = p6*der_act(max6[t][s]);
				w_up7[n] = p7*der_act(max7[t][s]);

				p00 = p00 + w_up0[n];
				p10 = p10 + w_up1[n];
				p20 = p20 + w_up2[n];
				p30 = p30 + w_up3[n];
				p40 = p40 + w_up4[n];
				p50 = p50 + w_up5[n];
				p60 = p60 + w_up6[n];
				p70 = p70 + w_up7[n];


			}
			b000 = b00 - e*p00;
			b010 = b01 - e* p10;
			b020 = b02 - e* p20;
			b030 = b03 - e* p30;
			b040 = b04 - e* p40;
			b050 = b05 - e* p50;
			b060 = b06 - e* p60;
			b070 = b07 - e* p70;


			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					double w000 = 0;
					double w100 = 0;
					double w200 = 0;
					double w300 = 0;
					double w400 = 0;
					double w500 = 0;
					double w600 = 0;
					double w700 = 0;


					for (int n = 0; n < 144; n++)
					{
						int q = n / 12;
						int s = n % 12;
						int i30 = maxindex0[q][s] / 2;
						int j30 = maxindex0[q][s] % 2;
						int i31 = maxindex1[q][s] / 2;
						int j31 = maxindex1[q][s] % 2;
						int i32 = maxindex2[q][s] / 2;
						int j32 = maxindex2[q][s] % 2;
						int i33 = maxindex3[q][s] / 2;
						int j33 = maxindex3[q][s] % 2;
						int i34 = maxindex4[q][s] / 2;
						int j34 = maxindex4[q][s] % 2;
						int i35 = maxindex5[q][s] / 2;
						int j35 = maxindex5[q][s] % 2;
						int i36 = maxindex6[q][s] / 2;
						int j36 = maxindex6[q][s] % 2;
						int i37 = maxindex7[q][s] / 2;
						int j37 = maxindex7[q][s] % 2;
						//printf("%d\n%d\n",i30,j30);
						int p = 2 * s + j;
						int t = 2 * q + i;
						w000 = w000 + dataarr[t + i30][p + j30] * w_up0[n];
						w100 = w100 + dataarr[t + i31][p + j31] * w_up1[n];
						w200 = w200 + dataarr[t + i32][p + j32] * w_up2[n];
						w300 = w300 + dataarr[t + i33][p + j33] * w_up3[n];
						w400 = w400 + dataarr[t + i34][p + j34] * w_up4[n];
						w500 = w500 + dataarr[t + i35][p + j35] * w_up5[n];
						w600 = w600 + dataarr[t + i36][p + j36] * w_up6[n];
						w700 = w700 + dataarr[t + i37][p + j37] * w_up7[n];	


					}
					w00[i][j] = w0[i][j] - e* w000;
					w10[i][j] = w1[i][j] - e* w100;
					w20[i][j] = w2[i][j] - e* w200;
					w30[i][j] = w3[i][j] - e*w300;
					w40[i][j] = w4[i][j] - e* w400;
					w50[i][j] = w5[i][j] - e*w500;
					w60[i][j] = w4[i][j] - e* w600;
					w70[i][j] = w5[i][j] - e*w700;


				}
			}

			//new weights
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					w0[i][j] = w00[i][j];
					w1[i][j] = w10[i][j];
					w2[i][j] = w20[i][j];
					w3[i][j] = w30[i][j];
					w4[i][j] = w40[i][j];
					w5[i][j] = w50[i][j];
					w6[i][j] = w60[i][j];
					w7[i][j] = w70[i][j];



				}
			}

			b00 = b000;
			b01 = b010;
			b02 = b020;
			b03 = b030;
			b04 = b040;
			b05 = b050;
			b06 = b060;
			b07 = b070;


			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					u0[i][j] = u00[i][j];
					u1[i][j] = u10[i][j];
					u2[i][j] = u20[i][j];
					u3[i][j] = u30[i][j];
					u4[i][j] = u40[i][j];
					u5[i][j] = u50[i][j];
					u6[i][j] = u60[i][j];
					u7[i][j] = u70[i][j];

				}
			}
			for (int i = 0; i < 44; i++)
			{
				b10[i] = b100[i];
				for (int j = 0; j < 10; j++)
				{
					v0[i][j] = v00[i][j];
				}
			}

			for (int i = 0; i < 10; i++)
			{
				b20[i] = b200[i];
			}
			//printf("running %d\n", r); 
		}
		//VALIDATION
		int trainTrue = 0;
		
		int testTrue = 0;
	
		//printf("start epoch 1\n");
		
		for (int r = 0; r < 60000; r++)
		{
			for (int k = 0; k < 28; k++)
				for (int l = 0; l < 28; l++)
					dataarr[k][l] = dataarr1[r][l + k * 28];

			

			//Convolution layer calculation--//forward propagation

			for (int i = 0; i < 24; i++)
			{
				for (int j = 0; j < 24; j++)
				{
					y00[i][j] = 0;
					y01[i][j] = 0;
					y02[i][j] = 0;
					y03[i][j] = 0;
					y04[i][j] = 0;
					y05[i][j] = 0;
					y06[i][j] = 0;
					y07[i][j] = 0;

					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							int p = i + k;
							int s = j + l;
							y00[i][j] = y00[i][j] + w0[k][l] * dataarr[p][s];
							y01[i][j] = y01[i][j] + w1[k][l] * dataarr[p][s];
							y02[i][j] = y02[i][j] + w2[k][l] * dataarr[p][s];
							y03[i][j] = y03[i][j] + w3[k][l] * dataarr[p][s];
							y04[i][j] = y04[i][j] + w4[k][l] * dataarr[p][s];
							y05[i][j] = y05[i][j] + w5[k][l] * dataarr[p][s];
							y06[i][j] = y06[i][j] + w6[k][l] * dataarr[p][s];
							y07[i][j] = y07[i][j] + w7[k][l] * dataarr[p][s];

						}

					}
					y00[i][j] = y00[i][j] + b00;
					y01[i][j] = y01[i][j] + b01;
					y02[i][j] = y02[i][j] + b02;
					y03[i][j] = y03[i][j] + b03;
					y04[i][j] = y04[i][j] + b04;
					y05[i][j] = y05[i][j] + b05;
					y06[i][j] = y06[i][j] + b06;
					y07[i][j] = y07[i][j] + b07;

					y00s[i][j] = act(y00[i][j]);
					y01s[i][j] = act(y01[i][j]);
					y02s[i][j] = act(y02[i][j]);
					y03s[i][j] = act(y03[i][j]);
					y04s[i][j] = act(y04[i][j]);
					y05s[i][j] = act(y05[i][j]);
					y06s[i][j] = act(y06[i][j]);
					y07s[i][j] = act(y07[i][j]);


				}
			}


			//MAX POOLING
			for (int i = 0; i < 12; i++)
			{
				for (int j = 0; j < 12; j++)
				{
					max0[i][j] = y00s[2 * i][2 * j];
					max1[i][j] = y01s[2 * i][2 * j];
					max2[i][j] = y02s[2 * i][2 * j];
					max3[i][j] = y03s[2 * i][2 * j];
					max4[i][j] = y04s[2 * i][2 * j];
					max5[i][j] = y05s[2 * i][2 * j];
					max6[i][j] = y06s[2 * i][2 * j];
					max7[i][j] = y07s[2 * i][2 * j];

					
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							int p = 2 * i + k;
							int s = 2 * j + l;
							if (max0[i][j] < y00s[p][s])
							{
								max0[i][j] = y00s[p][s];
								
							}
							if (max1[i][j] < y01s[p][s])
							{
								max1[i][j] = y01s[p][s];
								
							}
							if (max2[i][j] < y02s[p][s])
							{
								max2[i][j] = y02s[p][s];
								
							}
							if (max3[i][j] < y03s[p][s])
							{
								max3[i][j] = y03s[p][s];
								
							}
							if (max4[i][j] < y04s[p][s])
							{
								max4[i][j] = y04s[p][s];
								
							}
							if (max5[i][j] < y05s[p][s])
							{
								max5[i][j] = y05s[p][s];
								
							}
							if (max6[i][j] < y06s[p][s])
							{
								max6[i][j] = y06s[p][s];
								
							}
							if (max7[i][j] < y07s[p][s])
							{
								max7[i][j] = y07s[p][s];
								
							}

						}

					}


				}
			}


			
			//Fully connected layer
			for (int i = 0; i < 44; i++)
			{
				y10[i] = 0;
				for (int k = 0; k < 144; k++)
				{
					int p = k / 12;
					int q = k % 12;
					y10[i] = y10[i] + max0[p][q] * u0[k][i]
						+ max1[p][q] * u1[k][i]
						+ max2[p][q] * u2[k][i]
						+ max3[p][q] * u3[k][i]
						+ max4[p][q] * u4[k][i]
						+ max5[p][q] * u5[k][i] + max6[p][q] * u6[k][i] + max7[p][q] * u7[k][i];

				}
				y10[i] = y10[i] + b10[i];
				y10s[i] = act(y10[i]);
				




			}

			

			//OUTPUT LAYER
			for (int i = 0; i < 10; i++)
			{
				y20[i] = 0;
				for (int k = 0; k < 44; k++)
				{
					y20[i] = y20[i] + y10s[k] * v0[k][i];
				}
				y20[i] = y20[i] + b20[i];
				z[i] = act(y20[i]);

				

			}

				
				std::vector<int> vl;
				vl.resize(10);
				//double max = std::max_element(z, z + 10);
				int indmax= std::max_element(z.begin(), z.end()) - z.begin();
				for(int i=0;i<10;i++)
				{
					if(i==indmax)
						vl[i]=1;
					else
						vl[i]=0;
						
				}
				if (r < 10)
				{
					for (int j = 0; j < 10; j++) {
						printf("%d", vl[j]);
						
					}printf("  ");
					for (int j = 0; j < 10; j++) {
						printf("%d", labels[r][j]);

					}
					    
				} if (r  <10) printf("\n");
				if (labels[r] == vl)
					trainTrue++;

			}

			double accuracytrain = double(trainTrue) / 60000.0 * 100.0;
			//TESTING
			for (int r = 0; r < 10000; r++)
			{
				for (int k = 0; k < 28; k++)
					for (int l = 0; l < 28; l++)
						dataarr[k][l] = testarr1[r][l + k * 28];



				//Convolution layer calculation--//forward propagation

				for (int i = 0; i < 24; i++)
				{
					for (int j = 0; j < 24; j++)
					{
						y00[i][j] = 0;
						y01[i][j] = 0;
						y02[i][j] = 0;
						y03[i][j] = 0;
						y04[i][j] = 0;
						y05[i][j] = 0;
						y06[i][j] = 0;
						y07[i][j] = 0;

						for (int k = 0; k < 5; k++)
						{
							for (int l = 0; l < 5; l++)
							{
								int p = i + k;
								int s = j + l;
								y00[i][j] = y00[i][j] + w0[k][l] * dataarr[p][s];
								y01[i][j] = y01[i][j] + w1[k][l] * dataarr[p][s];
								y02[i][j] = y02[i][j] + w2[k][l] * dataarr[p][s];
								y03[i][j] = y03[i][j] + w3[k][l] * dataarr[p][s];
								y04[i][j] = y04[i][j] + w4[k][l] * dataarr[p][s];
								y05[i][j] = y05[i][j] + w5[k][l] * dataarr[p][s];
								y06[i][j] = y06[i][j] + w6[k][l] * dataarr[p][s];
								y07[i][j] = y07[i][j] + w7[k][l] * dataarr[p][s];

							}

						}
						y00[i][j] = y00[i][j] + b00;
						y01[i][j] = y01[i][j] + b01;
						y02[i][j] = y02[i][j] + b02;
						y03[i][j] = y03[i][j] + b03;
						y04[i][j] = y04[i][j] + b04;
						y05[i][j] = y05[i][j] + b05;
						y06[i][j] = y06[i][j] + b06;
						y07[i][j] = y07[i][j] + b07;

						y00s[i][j] = act(y00[i][j]);
						y01s[i][j] = act(y01[i][j]);
						y02s[i][j] = act(y02[i][j]);
						y03s[i][j] = act(y03[i][j]);
						y04s[i][j] = act(y04[i][j]);
						y05s[i][j] = act(y05[i][j]);
						y06s[i][j] = act(y06[i][j]);
						y07s[i][j] = act(y07[i][j]);


					}
				}


				//MAX POOLING
				for (int i = 0; i < 12; i++)
				{
					for (int j = 0; j < 12; j++)
					{
						max0[i][j] = y00s[2 * i][2 * j];
						max1[i][j] = y01s[2 * i][2 * j];
						max2[i][j] = y02s[2 * i][2 * j];
						max3[i][j] = y03s[2 * i][2 * j];
						max4[i][j] = y04s[2 * i][2 * j];
						max5[i][j] = y05s[2 * i][2 * j];
						max6[i][j] = y06s[2 * i][2 * j];
						max7[i][j] = y07s[2 * i][2 * j];


						for (int k = 0; k < 2; k++)
						{
							for (int l = 0; l < 2; l++)
							{
								int p = 2 * i + k;
								int s = 2 * j + l;
								if (max0[i][j] < y00s[p][s])
								{
									max0[i][j] = y00s[p][s];

								}
								if (max1[i][j] < y01s[p][s])
								{
									max1[i][j] = y01s[p][s];

								}
								if (max2[i][j] < y02s[p][s])
								{
									max2[i][j] = y02s[p][s];

								}
								if (max3[i][j] < y03s[p][s])
								{
									max3[i][j] = y03s[p][s];

								}
								if (max4[i][j] < y04s[p][s])
								{
									max4[i][j] = y04s[p][s];

								}
								if (max5[i][j] < y05s[p][s])
								{
									max5[i][j] = y05s[p][s];

								}
								if (max6[i][j] < y06s[p][s])
								{
									max6[i][j] = y06s[p][s];

								}
								if (max7[i][j] < y07s[p][s])
								{
									max7[i][j] = y07s[p][s];

								}

							}

						}


					}
				}



				//Fully connected layer
				for (int i = 0; i < 44; i++)
				{
					y10[i] = 0;
					for (int k = 0; k < 144; k++)
					{
						int p = k / 12;
						int q = k % 12;
						y10[i] = y10[i] + max0[p][q] * u0[k][i]
							+ max1[p][q] * u1[k][i]
							+ max2[p][q] * u2[k][i]
							+ max3[p][q] * u3[k][i]
							+ max4[p][q] * u4[k][i]
							+ max5[p][q] * u5[k][i] + max6[p][q] * u6[k][i] + max7[p][q] * u7[k][i];

					}
					y10[i] = y10[i] + b10[i];
					y10s[i] = act(y10[i]);





				}



				//OUTPUT LAYER
				for (int i = 0; i < 10; i++)
				{
					y20[i] = 0;
					for (int k = 0; k < 44; k++)
					{
						y20[i] = y20[i] + y10s[k] * v0[k][i];
					}
					y20[i] = y20[i] + b20[i];
					z[i] = act(y20[i]);



				}


				std::vector<int> vl;
				vl.resize(10);
				//double max = std::max_element(z, z + 10);
				int indmax = std::max_element(z.begin(), z.end()) - z.begin();
				for (int i = 0; i<10; i++)
				{
					if (i == indmax)
						vl[i] = 1;
					else
						vl[i] = 0;

				}
				
				if (testlabel[r] == vl)
					testTrue++;

			}

			double accuracytest = double(testTrue) / 10000.0 * 100.0;
			printf("training accuracy= %lf\n", accuracytrain);
			printf("test accuracy= %lf\n" , accuracytest );
			printf("end OF EPOCH %d\n\n", d3 + 1);
			outfile << "training accuracy= " << accuracytrain << endl;
			outfile << "test accuracy= " << accuracytest << endl;
			
			outfile << "weigths after epoch " << d3 + 1 << " are" << endl;
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w0[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w1[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w2[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w3[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w4[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w5[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w6[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					outfile << w7[i][j] << ",";
				}outfile << endl;
			}
			
			outfile << b00 << endl;
			
			outfile << b01 << endl;
			
			outfile << b02 << endl;
			
			outfile << b03 << endl;
			
			outfile << b04 << endl;
			
			outfile << b05 << endl;
			
			outfile << b06 << endl;
			
			outfile << b07 << endl;


			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u0[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u1[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u2[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u3[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u4[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u5[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u6[i][j] << ",";
				}outfile << endl;
			}
			for (int i = 0; i < 144; i++)
			{
				for (int j = 0; j < 44; j++)
				{
					outfile << u7[i][j] << ",";
				}outfile << endl;
			}

			for (int i = 0; i < 44; i++)
			{
				
				outfile  << b10[i] << endl;

			}


			for (int i = 0; i < 44; i++)
			{
				for (int j = 0; j < 10; j++)
				{
					
					outfile << v0[i][j] <<"," ;

				}outfile << endl;
			}

			for (int j = 0; j < 10; j++)
			{
				
				outfile  << b20[j] << endl;
			}
			outfile << "end OF EPOCH " << d3 + 1 << endl << endl;
		}

	getchar();
	return 0;

}


