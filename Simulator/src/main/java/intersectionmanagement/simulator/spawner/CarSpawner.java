package intersectionmanagement.simulator.spawner;

import intersectionmanagement.simulator.Actor;
import intersectionmanagement.simulator.Simulator;
import intersectionmanagement.simulator.car.Car;
import intersectionmanagement.simulator.control.NeuralNetworkController;
import intersectionmanagement.simulator.track.Node;

import java.util.ArrayList;

public class CarSpawner extends Actor {


    public enum Function {LINEAR, CONSTANT, SIN}

    public int simulationSteps;
    public int steps;
    public int counter;
    public byte[] serializedNetwork;
    public Function function;
    public double[] params;
    public double randomness;

    public CarSpawner(Simulator simulator, Node spawnLocation, byte[] serializedNetwork, int simulationSteps, Function function, double[] params, double randomness) {
        super(simulator, spawnLocation);
        speed = 0f;
        radius = 0f;

        this.simulationSteps = simulationSteps;
        this.steps = 0;
        this.function = function;
        this.params = params;
        counter = simulator.getRNG().nextInt(getRate());
        this.serializedNetwork = serializedNetwork;
        this.randomness = randomness;
    }

    @Override
    public void step(ArrayList<Actor> actorArray) {
        counter--;
        if (counter <= 0) {
            int rate = getRate();
            counter = rate + simulator.getRNG().nextInt((int) (rate*randomness));
            simulator.addCar(new Car(simulator, target, new NeuralNetworkController(serializedNetwork)));
        }
        if (steps < simulationSteps) {
            steps++;
        }
    }

    public int getRate() {
        switch (function) {
            case LINEAR:
                return linear();
            case CONSTANT:
                return constant();
            case SIN:
                return sin();
            default:
                throw new RuntimeException("No function specified for spawner");
        }
    }

    public int sin() {
        // params
        // 0 - period multiplier
        // 1 - min period
        // 2 - max period
        return (int) (params[1] + (1 - (0.5 + 0.5*Math.sin(params[0]*Math.PI*2*(steps*1.0)/simulationSteps)))*(params[2]-params[1]));
    }

    public int constant() {
        // params
        // 0 - rate
        return (int) params[0];
    }

    public int linear() {
        // params
        // 0 - min period
        // 1 - max period
        return (int) (params[0] + (1 - (steps*1.0)/simulationSteps)*(params[1]-params[0]));
    }
}
