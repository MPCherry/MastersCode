package intersectionmanagement.simulator.control;

import intersectionmanagement.simulator.Utility;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.networks.BasicNetwork;

public class HeuristicController implements CarController {
    @Override
    public double[] getControls(double[] sensors) {
        double[] controls = {1.f, 0.5f};
        if (sensors[0] > 0.9) {
            controls[0] = (float) (sensors[1]/Utility.CAR_SPEED_MAX);
        }
        return controls;
    }

    @Override
    public NEATNetwork getNEATNetwork() {
        return null;
    }

    @Override
    public BasicNetwork getBasicNetwork() {
        return null;
    }
}
