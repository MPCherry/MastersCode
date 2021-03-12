package intersectionmanagement.simulator.pedestrian;

import intersectionmanagement.simulator.Actor;
import intersectionmanagement.simulator.Simulator;
import intersectionmanagement.simulator.Utility;
import intersectionmanagement.simulator.track.Node;

public class Pedestrian extends Actor {
    public Pedestrian(Simulator simulator, Node firstTarget) {
        super(simulator, firstTarget);
        speed = Utility.PEDESTRIAN_SPEED;
        radius = 0.5f;
        solid = true;
    }
}
