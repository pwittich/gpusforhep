#pragma once

#include <stdio.h>
#include <math.h>

struct SGauss
{
	double A; //amplitude
	double m; //mean
	double o; //dispersion
};

class CHisto
{
public:
	CHisto(void);
	~CHisto(void);

	void	init(int numOfBins, double binStart, double binSz);

	void	initByRange(double binMin, double binMax, double binSz);

	void	clear();

	bool	binValue(double v,FILE *fp=0);

	bool	fitGauss(double &mean, double &amp, double &stdDev);

	bool	fitGauss();

	int		getBin(int i) {if(!hist || i < 0 || i >= nob) return -1; return hist[i];}

	int		getMaxBinIndex() {if(!hist) return -1; return maxBin;}

	int		getMaxBinValue() {if(!hist) return -1; return histMaxVal;}

	void	getBookends(int &front, int &back) {front = bookEnds[0]; back = bookEnds[1];}

	int		getNumOfBins() {if(!hist) return -1; return nob;}

	int		getNumOfEntries() {if(!hist) return -1; int sum=0; for(int i=0;i<nob;++i) sum += hist[i]; return sum;}

	double	getBinStart() {return bst;}

	double	getBinSize() {return bsz;}

	double	getGaussMeanBin() {return fg.m;}

	double	getGaussMeanValue() {return bst + fg.m*bsz + bsz/2;}

	double	getFWHM() {if(!fitDone) return 0; return 2.35482*fg.o/sqrt(2.0f)*bsz;}

	double	getFWHMinBinUnits() {if(!fitDone) return 0; return 2.35482*fg.o/sqrt(2.0f);}

	int		setBin(int b, int hv);

	void	setBinStart(double binStart) {bst = binStart;}

	void	setBinSize(double binSize) {bsz = binSize;}

	bool	save(FILE *fp);


private:

	double	FitCost(SGauss *g);
	double	Refine(SGauss *g, double &rv);


		//error histogram params
	int			*hist;
	int			nob;
	double		bsz,bst;

	int			histMaxVal;
	int			maxBin;
	int			bookEnds[2]; //define the bin with the first and last pieces of non-zero info

	bool		fitDone;
	SGauss		fg; //gaussian fit in terms of bins
};
