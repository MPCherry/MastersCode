package intersectionmanagement.simulator.control;

import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.networks.BasicNetwork;

public interface CarController {

    double[] getControls(double[] sensors);

    NEATNetwork getNEATNetwork();

    BasicNetwork getBasicNetwork();
}
