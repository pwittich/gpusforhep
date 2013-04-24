#include "histo.h"

CHisto::CHisto(void)
{
	hist = 0;
	histMaxVal = 0;
	maxBin = 0;
	bookEnds[0] = -1; bookEnds[1] = -1;
	fitDone = false;
}

CHisto::~CHisto(void)
{
	if(hist)
		delete[] hist;
}

void CHisto::initByRange(double binMin, double binMax, double binSz)
{
	init(int((binMax-binMin)/binSz),binMin,binSz);
}

void CHisto::init(int numOfBins, double binStart, double binSz)
{
	if(numOfBins < 0 || binSz <= 0) return;

	nob = numOfBins;
	bsz = binSz;
	bst = binStart;

		//init histogram structure
	if(hist)
		delete[] hist;

	hist = new int[nob];

	clear();
}

void CHisto::clear()
{
	for(int i=0;i<nob;++i)
		hist[i] = 0;

	histMaxVal = 0;
	maxBin = 0;
	bookEnds[0] = -1; bookEnds[1] = -1;
	fitDone = false;
}

//bookends are maintained here. bookEnds[0] is first bin with data
//bookEnds[1] is last bin with data.
bool CHisto::binValue(double v,FILE *fp)
{
	if(!hist) return false;

	int b = int((v - bst)/bsz);

	if(b < 0) b = 0;
	if(b >= nob) b = nob - 1;


	if(++hist[b] > histMaxVal) //check for new max value
	{
		histMaxVal = hist[b];
		maxBin = b;
	}

		//update bookends
	if(bookEnds[0] == -1)
		bookEnds[0] = bookEnds[1] = b;
	else
	{
		if(b < bookEnds[0])
			bookEnds[0] = b;
		if(b >  bookEnds[1])
			 bookEnds[1] = b;
	}

	if(fp)
	{
		//if(v < bst + bsz*b || v >= bst + bsz*(b+1))
			fprintf(fp,"bin %d %20.10f %d \t\t leftBound %f rightBound %f \n",b,v,hist[b],bst + bsz*b,bst + bsz*(b+1));
	}

	return true;
}

int	CHisto::setBin(int b, int hv)
{
	if(!hist || b < 0 || b >= nob) return -1;

	hist[b] = hv;

	if(hist[b] > histMaxVal) //check for new max value
	{
		histMaxVal = hist[b];
		maxBin = b;
	}

	if(hist[b]) //if non-zero
	{
			//update bookends
		if(bookEnds[0] == -1)
			bookEnds[0] = bookEnds[1] = b;
		else
		{
			if(b < bookEnds[0])
				bookEnds[0] = b;
			if(b >  bookEnds[1])
				 bookEnds[1] = b;
		}
	}

	return hist[b];
}


bool CHisto::fitGauss(double &mean, double &amp, double &stdDev)
{
	if(!fitGauss()) return false;

	mean = fg.m;
	amp = fg.A;		//fg.A is defined as 1/(stdDev*sqrt(2*PI))
	stdDev = fg.o/sqrt(double(2)); //fg.o is defined as sqrt(2*var).. stdDev = sqrt(var);

	return true;
}

bool CHisto::fitGauss()
{
	if(!hist || bookEnds[0] == -1) return false; //no data

	SGauss temp;

	//fill A,m,o with initial gauss parameter guess
	fg.A = double(hist[maxBin]); //max value
	fg.m = (double)maxBin;
	fg.o = (bookEnds[1]-bookEnds[0]+1)/4.0f;

	double refCost,currCost;
	bool isImproved = true;

	if(bookEnds[1]-bookEnds[0] <= 1) //at most 2 bins used.. so just use weighted avg
	{
		fg.m = (bookEnds[0]*hist[bookEnds[0]] + bookEnds[1]*hist[bookEnds[1]])*1.0f/
			(hist[bookEnds[0]] + hist[bookEnds[1]]);
		goto WRAP_UP;
	}

	currCost = FitCost(&fg);
	isImproved = true;

		//Refine
	while(isImproved)
	{
		isImproved = false;

		temp = fg;
		if( (refCost = Refine(&temp,temp.o)) + 0.00001f < currCost )	{ fg = temp; currCost = refCost; isImproved = true; }

		temp = fg;
		if( (refCost = Refine(&temp,temp.m)) + 0.00001f < currCost )	{ fg = temp; currCost = refCost; isImproved = true; }

		temp = fg;
		if( (refCost = Refine(&temp,temp.A)) + 0.00001f < currCost )	{ fg = temp; currCost = refCost; isImproved = true; }
	}

WRAP_UP:
	fitDone = true;
	return true;
}

double CHisto::FitCost(SGauss *g)
{
	double c=0;

	for(int i=0;i<nob;i++)
		c += (double)(fabs(hist[i] - (double)(g->A * exp(	-((i-g->m)/g->o)*((i-g->m)/g->o) ))));

	return c;
}

	//Gaussian-Fit Refinement paramaters
#define INIT_STEP		0.1
#define MIN_STEP		0.001

double CHisto::Refine(SGauss *g, double &rv)
{
	double c, tc, step = (double)INIT_STEP;
	int dir;

		//find direction
	rv += step;
	c = FitCost(g);
	rv -= 2*step;
	tc = FitCost(g);
	rv += step;

	if(c < tc)
		dir = 1;
	else if(c > tc)
	{	dir = -1; c = tc;	}
	else
	{	rv -= step; return c;		}		//they're equal so doesn't matter

	rv += dir*step;

		//iterate
	while(step >= MIN_STEP)
	{

		rv += dir*step;
		while(c > (tc = FitCost(g)))
		{
			c = tc;
			rv += dir*step;
		}

			//the cost is either equal or has increased, so undo last move, change direction, and decrease step
		dir *= -1;
		rv += dir*step;
		step /= 2;
	}

	return c;
}

bool CHisto::save(FILE *fp)
{
	if(!fp) return false;

	fprintf(fp,"\nHisto: NumOfBins %d NumOfEntries %d BinStart %15.2f BinSz %15.2f\nBookends %d to %d -- Mean Bin %5.3f Mean Val %5.3f\n\n",
		nob,getNumOfEntries(),bst,bsz,bookEnds[0],bookEnds[1],fg.m,bst + fg.m*bsz + bsz/2);

	for(int b=0;b<nob;++b)
		fprintf(fp,"\tBin %5d start %15.2f end %15.2f -- val %d\n",b,bst+b*bsz,bst+(b+1)*bsz,hist[b]);

	return true;
}
