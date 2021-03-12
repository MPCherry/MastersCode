package intersectionmanagement.simulator.car;

import intersectionmanagement.simulator.Actor;
import intersectionmanagement.simulator.Utility;

import java.util.ArrayList;

public class Sensor {
    public float distance;
    public float angle;
    public float radius;
    public boolean lastActivated = false;
    public float lastSpeed = 0;

    Sensor(float distance, float angle, float radius) {
        this.distance = distance;
        this.angle = angle;
        this.radius = radius;
    }

    public float getX(Car parent) {
        return calculateX(parent.x, parent.direction);
    }

    public float getY(Car parent) {
        return calculateY(parent.y, parent.direction);
    }

    public boolean activated(Car parent, ArrayList<Actor> actorArray) {
        float x = calculateX(parent.x, parent.direction);
        float y = calculateY(parent.y, parent.direction);

        for (Actor actor : actorArray) {
            if (actor == parent) {
                continue;
            }
            if (!actor.solid) {
                continue;
            }
            if (Utility.distance(x, y, actor.x, actor.y) < this.radius + actor.radius) {
                lastActivated = true;
                lastSpeed = actor.speed;
                return true;
            }
        }
        lastActivated = false;
        return false;
    }

    public boolean getLastActivated() {
        return lastActivated;
    }

    public float calculateX(float anchorX, float anchorDirection) {
        return anchorX + (float) Math.cos(anchorDirection + angle) * distance;
    }

    public float calculateY(float anchorY, float anchorDirection) {
        return anchorY + (float) Math.sin(anchorDirection + angle) * distance;
    }
}
