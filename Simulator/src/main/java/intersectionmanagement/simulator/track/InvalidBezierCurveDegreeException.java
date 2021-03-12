package intersectionmanagement.simulator.track;

class InvalidBezierCurveDegreeException extends RuntimeException {
    InvalidBezierCurveDegreeException(int degree) {
        super(String.format("Invalid degree: %d", degree));
    }
}
