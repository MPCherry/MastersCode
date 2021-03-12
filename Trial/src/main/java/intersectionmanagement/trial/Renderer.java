package intersectionmanagement.trial;

import intersectionmanagement.simulator.Utility;
import intersectionmanagement.simulator.car.Car;
import intersectionmanagement.simulator.car.Sensor;
import intersectionmanagement.simulator.pedestrian.Pedestrian;
import intersectionmanagement.simulator.track.Node;
import org.apache.commons.lang3.SerializationUtils;
import org.encog.neural.neat.NEATLink;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.PixelFormat;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL11.GL_MODELVIEW;
import static org.lwjgl.opengl.GL11.glMatrixMode;

public class Renderer {
    private static float scale;
    private static float cameraX = 400;
    private static float cameraY = 400;

    private static Trial trial;
    private static ArrayList<Car> selectedActors;
    private static Car lastSelectedActor;

    private static NEATNetwork neatNetwork;
    private static BasicNetwork basicNetwork;
    private static ArrayList<Float> nodeXs;
    private static ArrayList<Float> nodeYs;

    static void setupWindow(String title, Trial trial, float scale, int width, int height, byte[] serializedNetwork) throws LWJGLException {
        Display.setDisplayMode(new DisplayMode(width, height));
        Display.create(new PixelFormat(8,0,0,8));
        Display.setTitle(String.format("IntersectionManagement - %s", title));
        glLoadIdentity(); // Resets any previous projection matrices
        glOrtho(0, width, height, 0, 1, -1);
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        Renderer.scale = scale;
        Renderer.trial = trial;
        selectedActors = new ArrayList<>();
        lastSelectedActor = null;

        try {
            neatNetwork = SerializationUtils.deserialize(serializedNetwork);
        } catch (ClassCastException e) {
            //System.out.println("Wasn't a NEAT network");
        }

        try {
            basicNetwork = SerializationUtils.deserialize(serializedNetwork);
        } catch (ClassCastException e) {
            //System.out.println("Wasn't a conventional network");
        }

        if (neatNetwork != null) {
            neatNetwork.getLinks();
        }
        if (basicNetwork != null) {
            basicNetwork.calculateNeuronCount();
        }

        nodeXs = new ArrayList<>(32);
        nodeYs = new ArrayList<>(32);
        initiateNodePositions();
    }

    private static void initiateNodePositions() {
        if (neatNetwork != null) {
            Random nodePositionGenerator = new Random();
            int numNodes = neatNetwork.getOutputIndex()+neatNetwork.getOutputCount();

            for (NEATLink link : neatNetwork.getLinks()) {
                if (link.getFromNeuron()+1 > numNodes) {
                    numNodes = link.getFromNeuron()+1;
                }
                if (link.getToNeuron()+1 > numNodes) {
                    numNodes = link.getToNeuron()+1;
                }
            }

            for (int i = 0; i < numNodes; i++) {
                nodeXs.add(0, 0f);
                nodeYs.add(0, 0f);
            }

            int inputs = neatNetwork.getInputCount()+1; // Add +1 for the bias gene
            int outputs = neatNetwork.getOutputCount();
            for (int i = 0; i < inputs; i++) {
                double y = i*(350.0/(inputs)) + 50;
                nodeXs.set(i, 450.0f);
                nodeYs.set(i, (float) y);
            }

            for (int i = neatNetwork.getOutputIndex()+neatNetwork.getOutputCount(); i < numNodes; i++) {
                nodeXs.set(i, nodePositionGenerator.nextInt(200)+500f);
                nodeYs.set(i, nodePositionGenerator.nextInt(350)+50f);
            }

            for (int i = 0; i < outputs+2; i++) {
                if (i == 0 || i == outputs+1) {
                    continue;
                }
                double y = i*(350.0/(outputs+2)) + 50;
                nodeXs.set(neatNetwork.getOutputIndex()+i-1, 750.0f);
                nodeYs.set(neatNetwork.getOutputIndex()+i-1, (float) y);
            }
        }
    }

    static void drawActors(ArrayList<Car> cars, ArrayList<Pedestrian> pedestrians, List<Node> track) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(binToFloat(0),binToFloat(43),binToFloat(54),1);

        List<Node> seenNodes = new LinkedList<>();
        List<Node> currentNodes = new LinkedList<>();
        List<Node> nextNodes = new LinkedList<>();
        currentNodes.addAll(track);

        while (!currentNodes.isEmpty()) {
            for (Node node : currentNodes) {
                for (Node nextNode : node.nextNodes) {
                    if (node.active) {
                        drawLine(node.x, node.y, nextNode.x, nextNode.y, 2, binToFloat(131), binToFloat(148), binToFloat(150), 1);
                        drawRoad(node.x, node.y, nextNode.x, nextNode.y);
                    }
                    else {
                        drawLine(node.x, node.y, nextNode.x, nextNode.y, 2, binToFloat(88), binToFloat(110), binToFloat(117), 1);
                        drawRoad(node.x, node.y, nextNode.x, nextNode.y);
                    }
                    nextNodes.add(nextNode);
                }
            }
            currentNodes.clear();
            currentNodes.addAll(nextNodes);
            currentNodes.removeAll(seenNodes);
            seenNodes.addAll(nextNodes);
            nextNodes.clear();
        }

        for (Car actor : cars) {
            float d = actor.direction;
            float[] actorX = {-2, -2, 2, 2};
            float[] actorY = {0.75f, -0.75f, -0.75f, 0.75f};
            for (int i = 0; i < 4; i++) {
                float x = actorX[i];
                float y = actorY[i];
                actorX[i] = actor.x + (float)(x*Math.cos(d) - y*Math.sin(d));
                actorY[i] = actor.y + (float)(y*Math.cos(d) + x*Math.sin(d));
            }
            if (actor == lastSelectedActor) {
                drawQuad(actorX, actorY, binToFloat(108), binToFloat(113), binToFloat(196), 1);
            } else {
                drawQuad(actorX, actorY, binToFloat(133), binToFloat(153), binToFloat(0), 1);
            }
            if (selectedActors.contains(actor)) {
                for (List<Sensor> sensorRow : actor.getSensors()) {
                    int count = 0;
                    for (Sensor sensor : sensorRow) {
                        count++;
                        if (count > 5) {
                            break;
                        }
                        if (sensor.getLastActivated()) {
                            drawCircle(20, sensor.getX(actor), sensor.getY(actor), sensor.radius, binToFloat(211), binToFloat(54), binToFloat(130), 0.2f);
                            break;
                        } else {
                            drawCircle(20, sensor.getX(actor), sensor.getY(actor), sensor.radius, binToFloat(42), binToFloat(161), binToFloat(152), 0.1f);
                        }
                    }

                }
            }
        }

        for (Pedestrian pedestrian : pedestrians) {
            drawCircle(20, pedestrian.x, pedestrian.y, pedestrian.radius, binToFloat(133), binToFloat(153), binToFloat(0), 1);
        }


        drawLine( 50 - 40, 20, 50 + 40, 20, 12, binToFloat(7), binToFloat(54), binToFloat(66), 1);
        drawLine( 50 - 40, 30, 50 + 40, 30, 12, binToFloat(7), binToFloat(54), binToFloat(66), 1);
        if (lastSelectedActor != null) {
            if (lastSelectedActor.targetSpeed >= 0) {
                drawLine(50, 20, 50 + lastSelectedActor.targetSpeed * 40, 20, 12, binToFloat(133), binToFloat(153), binToFloat(0), 1);
            } else {
                drawLine(50, 20, 50 + lastSelectedActor.targetSpeed * 40, 20, 12, binToFloat(220), binToFloat(50), binToFloat(47), 1);
            }

            if (lastSelectedActor.speed >= 0) {
                drawLine(50, 30, 50 + (lastSelectedActor.getTurnModifier()/Utility.CAR_TURN_MAX) * 40, 30, 12, binToFloat(133), binToFloat(153), binToFloat(0), 1);
            } else {
                drawLine(50, 30, 50 + (lastSelectedActor.getTurnModifier()/Utility.CAR_TURN_MAX) * 40, 30, 12, binToFloat(220), binToFloat(50), binToFloat(47), 1);
            }


            if (!cars.contains(lastSelectedActor)) {
                lastSelectedActor = null;
            }
        }

        Display.update();
        Display.sync(60);
    }


    private static void drawNeuralNetwork() {
        if (neatNetwork != null) {
            for (NEATLink link : neatNetwork.getLinks()) {
                int from = link.getFromNeuron();
                int to = link.getToNeuron();
                int r = 88;
                int g = 110;
                int b = 117;
                if (link.getWeight() > 0) {
                    r += (int) ((133-88)*(link.getWeight()/5));
                    g += (int) ((153-110)*(link.getWeight()/5));
                    b += (int) ((0-117)*(link.getWeight()/5));
                } else {
                    r += (int) ((220-88)*(link.getWeight()/-5));
                    g += (int) ((50-110)*(link.getWeight()/-5));
                    b += (int) ((47-117)*(link.getWeight()/-5));
                }
                drawLine(nodeXs.get(from), nodeYs.get(from), nodeXs.get(to), nodeYs.get(to), 2, binToFloat(r), binToFloat(g), binToFloat(b), 0.5f);
            }

            for (int i = 0; i < nodeXs.size(); i++) {
                drawCircle(16, nodeXs.get(i), nodeYs.get(i), 4, binToFloat(253), binToFloat(246), binToFloat(227), 1);
            }

        }

        if (basicNetwork != null) {
            for (int l = 0; l < basicNetwork.getLayerCount(); l++) {
                float x = l*(400.0f/basicNetwork.getLayerCount()) + 450;
                for (int n = 0; n < basicNetwork.getLayerNeuronCount(l)+2; n++) {
                    if (n == 0 || n == basicNetwork.getLayerNeuronCount(l)+1) {
                        continue;
                    }
                    float y = n*(350.0f/(basicNetwork.getLayerNeuronCount(l)+1)) + 25;
                    if (l < basicNetwork.getLayerCount()-1) {
                        for (int n2 = 0; n2 < basicNetwork.getLayerNeuronCount(l + 1) + 2; n2++) {
                            if (n2 == 0 || n2 == basicNetwork.getLayerNeuronCount(l + 1) + 1) {
                                continue;
                            }
                            if (basicNetwork.isConnected(l, n - 1, n2 - 1)) {
                                double weight = basicNetwork.getWeight(l, n - 1, n2 - 1);
                                int r = 88;
                                int g = 110;
                                int b = 117;
                                if (weight > 0) {
                                    r += (int) ((133 - 88) * (weight / 5));
                                    g += (int) ((153 - 110) * (weight / 5));
                                    b += (int) ((0 - 117) * (weight / 5));
                                } else {
                                    r += (int) ((220 - 88) * (weight / -5));
                                    g += (int) ((50 - 110) * (weight / -5));
                                    b += (int) ((47 - 117) * (weight / -5));
                                }
                                float xn2 = (l + 1) * (400.0f / basicNetwork.getLayerCount()) + 450;
                                float yn2 = (n2 * (350.0f / (basicNetwork.getLayerNeuronCount(l + 1) + 1)) + 25);
                                drawLine(x, y, xn2, yn2, 2, binToFloat(r), binToFloat(g), binToFloat(b), 0.5f);
                            }
                        }
                    }

                    drawCircle(16, x, y, 4, binToFloat(253), binToFloat(246), binToFloat(227), 1);
                }
            }
        }
    }

    private static boolean spaceDown = false;
    private static boolean leftMouseDown = false;

    static void handleInput(ArrayList<Car> actors) {
        if (Keyboard.isKeyDown(Keyboard.KEY_SPACE)) {
            if (!spaceDown) {
                trial.toggleSimulating();
                spaceDown = true;
            }
        } else {
            spaceDown = false;
        }

        if (Keyboard.isKeyDown(Keyboard.KEY_EQUALS)) {
            scale += 0.1;
            if (scale > 10) {
                scale = 10;
            }
        }

        if (Keyboard.isKeyDown(Keyboard.KEY_MINUS)) {
            scale -= 0.1;
            if (scale < 0.1) {
                scale = 0.1f;
            }
        }

        if (Keyboard.isKeyDown(Keyboard.KEY_UP)) {
            cameraY += 10;
        }
        if (Keyboard.isKeyDown(Keyboard.KEY_DOWN)) {
            cameraY -= 10;
        }
        if (Keyboard.isKeyDown(Keyboard.KEY_LEFT)) {
            cameraX += 10;
        }
        if (Keyboard.isKeyDown(Keyboard.KEY_RIGHT)) {
            cameraX -= 10;
        }

        if (Mouse.isButtonDown(0)) {
            if (!leftMouseDown) {
                int x = Mouse.getX();
                int y = Display.getHeight() - Mouse.getY();
                for (Car actor : actors) {
                    if (Utility.distance(x, y, center(actor.x, false), center(actor.y, true)) < actor.radius*scale) {
                        if (selectedActors.contains(actor)) {
                            selectedActors.remove(actor);
                        } else {
                            selectedActors.add(actor);
                            lastSelectedActor = actor;
                        }
                    }
                }
                leftMouseDown = true;
            }
        } else {
            leftMouseDown = false;
        }
    }

    private static void drawRoad(float x1, float y1, float x2, float y2) {
        float width = 3.7f/2;
        drawCircle(16, x1, y1, width, binToFloat(88), binToFloat(110), binToFloat(117), 1);
        drawCircle(16, x2, y2, width, binToFloat(88), binToFloat(110), binToFloat(117), 1);
        double d = Math.atan2(y2-y1, x2-x1);
        float xOffL = (float) (width*Math.cos(d+Math.PI/2));
        float xOffR = (float) (width*Math.cos(d-Math.PI/2));
        float yOffL = (float) (width*Math.sin(d+Math.PI/2));
        float yOffR = (float) (width*Math.sin(d-Math.PI/2));
        float[] x = {x1 + xOffL, x1 + xOffR, x2 + xOffR, x2 + xOffL};
        float[] y = {y1 + yOffL, y1 + yOffR, y2 + yOffR, y2 + yOffL};
        drawQuad(x, y, binToFloat(88), binToFloat(110), binToFloat(117), 1);
    }

    private static void drawCircle(int slices, float x, float y, float radius, float r, float g, float b, float a) {
        glColor4f(r, g, b, a);
        glBegin(GL_POLYGON);

        for (int i = 0; i < slices; i++)
        {
            double rad = (i/(slices*1.0))*2*Math.PI;
            glVertex2f(center(x, false) + (float) Math.cos(rad)*radius*scale, center(y, true) + (float)Math.sin(rad)*radius*scale);
        }

        glEnd();
    }

    private static void drawLine(float x1, float y1, float x2, float y2, float width, float r, float g, float b, float a) {
        glColor4f(r, g, b, a);
        glLineWidth(width);
        glBegin(GL_LINES);

        glVertex2f(center(x1, false), center(y1, true));
        glVertex2f(center(x2, false), center(y2, true));

        glEnd();
    }

    private static void drawQuad(float[] x, float[] y, float r, float g, float b, float a) {
        glColor4f(r, g, b, a);
        glBegin(GL_QUADS);

        for (int i = 0; i < 4; i++) {
            glVertex2f(center(x[i], false), center(y[i], true));
        }

        glEnd();
    }

    private static float binToFloat(int binary) {
        return binary/255.0f;
    }

    private static float center(float coord, boolean vertical) {
        if (vertical) {
            return scale * (coord - 200) + cameraY;
        } else {
            return scale * (coord - 200) + cameraX;
        }
    }
}
