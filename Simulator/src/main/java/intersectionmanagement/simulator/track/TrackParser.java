package intersectionmanagement.simulator.track;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import intersectionmanagement.simulator.Utility;
import org.json.JSONObject;
import org.json.JSONArray;

public class TrackParser {

    public static String getName(String trackPath) throws IOException {
        String trackFile = Utility.loadResource(trackPath);
        JSONObject trackJSON = new JSONObject(trackFile);

        return trackJSON.getString("name");
    }

    public static List<Node> parseTrack(String trackPath, boolean pedestrian) throws IOException {
        String trackFile = Utility.loadResource(trackPath);
        JSONObject trackJSON = new JSONObject(trackFile);

        List<Node> curves = new LinkedList<>();
        JSONArray curvesJSON = trackJSON.getJSONArray("curves");
        for (Object curveJSON : curvesJSON) {
            JSONObject curveData = (JSONObject) curveJSON;
            boolean curvePedestrian;
            if (!curveData.has("pedestrian")) {
                curvePedestrian = false;
            }
            else  {
                curvePedestrian = curveData.getBoolean("pedestrian");
            }

            if (curvePedestrian != pedestrian) {
                continue;
            }

            int degree = curveData.getInt("degree");
            int precision = 2;
            if (curveData.has("precision")) {
                precision = curveData.getInt("precision");
            }
            float[] x = new float[degree];
            float[] y = new float[degree];
            for (int i = 0; i < degree; i++) {
                x[i] = curveData.getFloat(String.format("x_%d", i));
                y[i] = curveData.getFloat(String.format("y_%d", i));
            }

            boolean active = curveData.getBoolean("active");
            curves.add(calculateBezierCurve(degree, precision, x, y, active));
        }

        curves = linkCurves(curves);
        return curves;
    }

    public static Node calculateBezierCurve(int degree, int precision, float[] x, float[] y, boolean active) {
        // Precision should specify number of points, since point has to be added at the end, use -1
        float increment = 1.0f/(precision-1);

        Node root = new Node(x[0], y[0], active);
        Node currentNode = root;

        for (float t = increment; t < 1; t += increment) {
            Node nextNode = calculateNode(degree, t, x, y, active);
            currentNode.nextNodes.add(nextNode);
            currentNode = nextNode;
        }

        currentNode.nextNodes.add(new Node(x[degree-1], y[degree-1], active));
        return root;
    }

    public static Node calculateNode(int degree, float t, float[] x, float[] y, boolean active) {
        float nextNodeX = bezierFunction(degree, t, x);
        float nextNodeY = bezierFunction(degree, t, y);
        return new Node(nextNodeX, nextNodeY, active);
    }

    public static float bezierFunction(int degree, float t, float[] points) {
        switch (degree) {
            case 2:
                return (1-t)*points[0]+t*points[1];
            case 3:
                return (float) (Math.pow(1-t, 2)*points[0] + 2*(1-t)*t*points[1] + Math.pow(t, 2)*points[2]);
            case 4:
                return (float) (
                        Math.pow(1-t, 3)*points[0] +
                                3*Math.pow(1-t, 2)*t*points[1] +
                                3*(1-t)*Math.pow(t, 2)*points[2] +
                                Math.pow(t, 3)*points[3]);
            default:
                throw new InvalidBezierCurveDegreeException(degree);
        }
    }

    /*
    Finds curves that have a start point equal to the end point of another curve and links them together
     */
    public static List<Node> linkCurves(List<Node> roots) {
        // If a root node is linked backwards, then it is no longer a root node, store these here
        List<Node> linkedNodes = new LinkedList<>();
        for (Node node : roots) {
            linkedNodes.addAll(recursiveLinkEnd(roots, node));
        }

        roots.removeAll(linkedNodes);
        return roots;
    }

    /*
    Recursively finds penultimate nodes and checks if their next nodes (which will be end nodes) are the same as any
    root nodes
    If so, then the penultimate node is linked to the root of the other curve and that curve's root node will be
    removed as a root node
     */
    public static List<Node> recursiveLinkEnd(List<Node> roots, Node node) {
        LinkedList<Node> removedEndNodes = new LinkedList<>();
        LinkedList<Node> addedEndNodes = new LinkedList<>();
        LinkedList<Node> removedRoots = new LinkedList<>();

        for (Node nextNode : node.nextNodes) {
            if (nextNode.nextNodes.size() == 0) {
                for (Node rootNode : roots) {
                    if (rootNode.x == nextNode.x && rootNode.y == nextNode.y) {
                        addedEndNodes.add(rootNode);
                        removedEndNodes.add(nextNode);
                        removedRoots.add(rootNode);
                    }
                }
            } else {
                removedRoots.addAll(recursiveLinkEnd(roots, nextNode));
            }
        }
        node.nextNodes.removeAll(removedEndNodes);
        node.nextNodes.addAll(addedEndNodes);
        return removedRoots;
    }

}
