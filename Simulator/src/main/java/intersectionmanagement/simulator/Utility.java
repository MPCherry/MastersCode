package intersectionmanagement.simulator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Utility {
    public static final float CAR_SPEED_MAX = 0.277f;
    public static final float CAR_SPEED_MIN = 0.0f;
    public static final float CAR_ACCELERATION = 0.0025f;
    public static final float CAR_BREAKING = 0.004f;
    public static final float CAR_TURN_MAX = 2.0f;

    public static final float PEDESTRIAN_SPEED = 0.04f;

    public static float distance(float x1, float y1, float x2, float y2) {
        return (float) Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2));
    }

    public static String loadResource(String file) throws IOException {
        InputStream in = Utility.class.getResourceAsStream("/"+file);
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        StringBuilder fileContent = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            fileContent.append(line).append("\n");
        }
        return fileContent.toString();
    }
}
