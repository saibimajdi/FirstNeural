// ConsoleApplicationCpp.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <strstream>
//#include <sstream>

using namespace std;


// ************* TrainningData **************
typedef std::vector<double> t_vals;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(t_vals &inputVals);
    unsigned getTargetOutputs(t_vals &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    std::string line;
    std::string label;

    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const std::string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(t_vals &inputVals)
{
    inputVals.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss>> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;

        while (ss >> oneValue)
            inputVals.push_back(oneValue);
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(t_vals &targetOutputVals)
{
    targetOutputVals.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;

        while (ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }

    return targetOutputVals.size();
}

// TrainingData
//
//


struct Connection
{
	double weight;
	double deltaWieght;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta;  // [0.0 .. 1.0]
	static double alpha // [0.0 .. n]
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;	
	unsigned m_myIndex;
	double m_gradient;
};
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWieght;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha + oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWieght = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	for(unsigned n=0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calculateOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::activationFunction(double x)
{
	return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}	

	m_outputVal = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < nu; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

// ****************** Network ******************
class Net
{
	public:
		Net(const vector<unsigned> &topology);
		void feedForward(const vector<double> &inputVals);
		void backPropagation(const vector<double> &targetVals) ;
		void getResults(vector<double> &resultVals) const {};

	private:
		vector<Layer> m_layers;
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();
	for(unsigned n =0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backPropagation(const vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1 ; ++n)
	{
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers

	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
	
		for(unsigned n =0;n <hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer, update connection weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1 ; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}

}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Added neuron! : " << neuronNum << endl;
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);
	for(unsigned i=0;i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagation
	for(unsigned layerNum = 1;layerNum < m_layers.size() ; ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n <m_layers[layerNum].size() - 1 ; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

int main()
{
	TrainingData trainData("trainsample/out_xor.txt");
    // TrainingData trainData("trainsample/out_and.txt");
    // TrainingData trainData("trainsample/out_or.txt");
    // TrainingData trainData("trainsample/out_no.txt");

    // e.g., { 3, 2, 1 }
    std::vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    t_vals inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) 
    {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass << std::endl;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0])
            break;

        showVectorVals("Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        std::cout << "Net current error: " << myNet.getError() << std::endl;
        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;

        if (trainingPass > 100 && myNet.getRecentAverageError() < 0.05)
        {
            std::cout << std::endl << "average error acceptable -> break" << std::endl;
            break;
        }
    }

    std::cout << std::endl << "Done" << std::endl;

    if (topology[0] == 2)
    {
        std::cout << "TEST" << std::endl;
        std::cout << std::endl;

        unsigned dblarr_test[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };

        for (unsigned i = 0; i < 4; ++i)
        {
            inputVals.clear();
            inputVals.push_back(dblarr_test[i][0]);
            inputVals.push_back(dblarr_test[i][1]);

            myNet.feedForward(inputVals);
            myNet.getResults(resultVals);

            showVectorVals("Inputs:", inputVals);
            showVectorVals("Outputs:", resultVals);

            std::cout << std::endl;
        }

        std::cout << "/TEST" << std::endl;
    }
}

