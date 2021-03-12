package intersectionmanagement.simulator;

import intersectionmanagement.simulator.track.Node;

import java.util.ArrayList;

public abstract class Actor {
    public float x;
    public float y;
    public float direction;
    public boolean finished;

    public Simulator simulator;
    public Node target;
    public Node currentNode;
    public boolean solid;
    public float wheelbase;

    public float speed;
    public float radius;

    public float targetSpeed;

    public Actor(Simulator simulator, Node firstTarget) {
        this.simulator = simulator;
        target = firstTarget;
        currentNode = firstTarget;
        x = target.x;
        y = target.y;
        finished = false;
        targetSpeed = 0;
        direction = (float) Math.atan2(target.nextNodes.get(0).y - y, target.nextNodes.get(0).x - x);
    }

    void moveTowardsTarget() {
        float targetAngle = (float) Math.atan2(target.y - y, target.x - x);
        float diffAngle = targetAngle - direction;
        float diffAngle2 = (targetAngle + (float) (Math.PI*2)) - direction;
        float diffAngle3 = (targetAngle - (float) (Math.PI*2)) - direction;
        float endDiffAngle = diffAngle;

        if (Math.abs(diffAngle2) < Math.abs(endDiffAngle)) {
            endDiffAngle = diffAngle2;
        }
        if (Math.abs(diffAngle3) < Math.abs(endDiffAngle)) {
            endDiffAngle = diffAngle3;
        }


        float turn;
        if (Math.abs(endDiffAngle) < Utility.CAR_TURN_MAX) {
            turn = endDiffAngle;
        } else {
            turn = (endDiffAngle/Math.abs(endDiffAngle)) * Utility.CAR_TURN_MAX;
        }

        turn += getTurnModifier();
        if (turn > Utility.CAR_TURN_MAX) {
            turn = Utility.CAR_TURN_MAX;
        } else if (turn < Utility.CAR_TURN_MAX*-1) {
            turn = -1*Utility.CAR_TURN_MAX;
        }

        float rearX = (float) (x - (wheelbase/2) * Math.cos(direction));
        float rearY = (float) (y - (wheelbase/2) * Math.sin(direction));
        float frontX = (float) (x + (wheelbase/2) * Math.cos(direction));
        float frontY = (float) (y + (wheelbase/2) * Math.sin(direction));

        rearX += (speed * Math.cos(direction));
        rearY += (speed * Math.sin(direction));

        frontX += (speed * Math.cos(direction+turn));
        frontY += (speed * Math.sin(direction+turn));

        x = (float) ((rearX+frontX)/2.0);
        y = (float) ((rearY+frontY)/2.0);

        direction = (float) Math.atan2(frontY-rearY, frontX-rearX);

        if (direction > Math.PI) {
            direction -= Math.PI*2;
        } else if (direction < -1*Math.PI) {
            direction += Math.PI*2;
        }

        if (distanceToTarget() < radius) {
            if (target.nextNodes.size() > 0) {
                int nextNode = simulator.getRNG().nextInt(target.nextNodes.size());
                currentNode = target;
                target = target.nextNodes.get(nextNode);
            } else {
                finished = true;
            }
        }
    }

    public void step(ArrayList<Actor> actorArray) {}

    public float getTurnModifier() {return 0;}

    public float distanceToTarget() {
        return Utility.distance(x, y, target.x, target.y);
    }

}
