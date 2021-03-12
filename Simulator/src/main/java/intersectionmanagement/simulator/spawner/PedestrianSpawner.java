package intersectionmanagement.simulator.spawner;

import intersectionmanagement.simulator.Actor;
import intersectionmanagement.simulator.Simulator;
import intersectionmanagement.simulator.pedestrian.Pedestrian;
import intersectionmanagement.simulator.track.Node;

import java.util.ArrayList;

public class PedestrianSpawner extends Actor {

    private int simulationSteps;
    private int steps;
    private int counter;
    private int pedestrianRate;
    private float pedestrianRandomness;
    private boolean pedestriansEnabled;

    public PedestrianSpawner(Simulator simulator, Node firstTarget, int simulationSteps, int pedestrianRate, float pedestrianRandomness) {
        super(simulator, firstTarget);
        speed = 0f;
        radius = 0f;

        this.simulationSteps = simulationSteps;
        this.pedestrianRate = pedestrianRate;
        this.pedestrianRandomness = pedestrianRandomness;

        pedestriansEnabled = (pedestrianRate != 0);
        if (pedestriansEnabled) {
            counter = pedestrianRate + simulator.getRNG().nextInt((int) (pedestrianRate * pedestrianRandomness));
        }
    }

    @Override
    public void step(ArrayList<Actor> actorArray) {
        counter--;
        if (pedestriansEnabled && counter <= 0) {
            int rate = pedestrianRate;
            counter = rate + simulator.getRNG().nextInt((int) (rate*pedestrianRandomness));
            simulator.addPedestrian(new Pedestrian(simulator, target));
        }
        if (steps < simulationSteps) {
            steps++;
        }
    }
}
