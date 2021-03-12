package intersectionmanagement.simulator.car;

import intersectionmanagement.simulator.Actor;
import intersectionmanagement.simulator.Simulator;
import intersectionmanagement.simulator.Utility;
import intersectionmanagement.simulator.control.CarController;
import intersectionmanagement.simulator.control.HeuristicController;
import intersectionmanagement.simulator.track.Node;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Car extends Actor {
    public CarController controller;
    public HeuristicController inactiveController;
    public boolean controllerActive;
    public List<List<Sensor>> sensorArray;
    public float turnControl;

    public Car(Simulator simulator, Node firstTarget, CarController controller) {
        super(simulator, firstTarget);
        speed = 0.5f;
        radius = 1.0f;
        solid = true;
        turnControl = 0;
        wheelbase = 4;

        this.controller = controller;
        this.inactiveController = new HeuristicController();
        controllerActive = false;
        sensorArray = new ArrayList<>();
        createSensorRow(0, 60, 24);
        createSensorRow((float) Math.PI, 30, 12);
        createSensorRow(-0.3f, 55, 16);
        createSensorRow(0.3f, 55, 16);
        createSensorRow(-0.65f, 50, 16);
        createSensorRow(0.65f, 50, 16);
        createSensorRow(-1.1f, 45, 16);
        createSensorRow(1.1f, 45, 16);
        createSensorRow((float)(0.5*Math.PI), 40, 16);
        createSensorRow((float)(-0.5*Math.PI), 40, 16);
        createSensorRow(2, 35, 12);
        createSensorRow(-2, 35, 12);
        createSensorRow(2.5f, 32, 12);
        createSensorRow(-2.5f, 32, 12);
    }

    @Override
    public void step(ArrayList<Actor> actorArray) {
        controllerActive = currentNode.active;


        double[] controls;
        if (controllerActive) {
            controls = controller.getControls(getSensorValues(actorArray));
        } else {
            controls = inactiveController.getControls(getSensorValues(actorArray, 1));
        }
        targetSpeed = (float) controls[0];
        turnControl = (float) ((controls[1]*2)-1)*(Utility.CAR_TURN_MAX*0.15f);

        float acceleration = limitAcceleration(targetSpeed*Utility.CAR_SPEED_MAX - speed);

        speed += acceleration;
        limitSpeed();
    }

    public void limitSpeed() {
        if (speed > Utility.CAR_SPEED_MAX) {
            speed = Utility.CAR_SPEED_MAX;
        }

        if (speed < Utility.CAR_SPEED_MIN) {
            speed = Utility.CAR_SPEED_MIN;
        }
    }

    public float limitAcceleration(float acceleration) {
        if (acceleration > Utility.CAR_ACCELERATION) {
            acceleration = Utility.CAR_ACCELERATION;
        }

        if (acceleration < -1 * Utility.CAR_BREAKING) {
            acceleration = -1 * Utility.CAR_BREAKING;
        }

        return acceleration;
    }

    @Override
    public float getTurnModifier() {
        return turnControl;
    }

    public void createSensorRow(float angle, float distance, int quantity) {
        LinkedList<Sensor> sensorRow = new LinkedList<>();
        for (int i = 0; i < quantity; i++) {
            // Add one to i because we want sensors starting off the car and ending at final distance
            float sensorDistance = (distance / quantity) * (i + 1);
            sensorRow.add(new Sensor(sensorDistance, angle, sensorDistance*0.15f));
        }
        sensorArray.add(sensorRow);
    }

    public List<List<Sensor>> getSensors() {
        return sensorArray;
    }

    public double[] getSensorValues(ArrayList<Actor> actorArray) {
        return getSensorValues(actorArray, sensorArray.size());
    }

    public double[] getSensorValues(ArrayList<Actor> actorArray, int maxSensors) {
        double[] sensorValues;
        if (maxSensors == 1) {
            sensorValues = new double[2];
        } else {
            sensorValues = new double[maxSensors];
        }
        for (int i = 0; i < maxSensors; i++) {
            List<Sensor> sensorRow = sensorArray.get(i);
            for (Sensor sensor : sensorRow) {
                if (sensor.activated(this, actorArray)) {
                    float value = 1 - ((sensorRow.indexOf(sensor)*1.0f) / sensorRow.size());
                    sensorValues[i] = value;
                    if (maxSensors == 1) {
                        sensorValues[1] = sensor.lastSpeed;
                    }
                    break;
                }
            }
        }
        return sensorValues;
    }
}
