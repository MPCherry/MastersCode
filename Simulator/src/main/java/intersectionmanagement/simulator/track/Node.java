package intersectionmanagement.simulator.track;

import java.util.List;
import java.util.ArrayList;

public class Node {
    public List<Node> nextNodes;
    public float x;
    public float y;
    public boolean active;

    Node(float x, float y, boolean active) {
        this.x = x;
        this.y = y;
        this.active = active;
        nextNodes = new ArrayList<>();
    }
}