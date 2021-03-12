package intersectionmanagement.simulator.control;

import org.apache.commons.lang3.SerializationUtils;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.networks.BasicNetwork;

public class NeuralNetworkController implements CarController {

    public NEATNetwork neatNetwork;
    public BasicNetwork basicNetwork;
    public MLRegression neuralNetwork;

    public NeuralNetworkController(byte[] serializedNetwork) {
        try {
            neatNetwork = SerializationUtils.deserialize(serializedNetwork);
            neuralNetwork = neatNetwork;
        } catch (ClassCastException e) {
            //System.out.println("Not NEAT");
        }
        try {
            basicNetwork = SerializationUtils.deserialize(serializedNetwork);
            neuralNetwork = basicNetwork;
        } catch (ClassCastException e) {
            //System.out.println("Not basic");
        }
    }

    @Override
    public double[] getControls(double[] sensors) {
        MLData inputData = new BasicMLData(sensors.length);
        inputData.setData(sensors);
        MLData outputData = neuralNetwork.compute(inputData);
        return outputData.getData();
//        double[] data = {1, 0.5};
//        return data;
    }

    @Override
    public NEATNetwork getNEATNetwork() {
        return neatNetwork;
    }

    @Override
    public BasicNetwork getBasicNetwork() {
        return basicNetwork;
    }
}
