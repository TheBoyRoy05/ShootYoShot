import { COURT_DIMENSIONS } from './courtConstants';

// Helper functions to create zone paths
export const createZonePath = (zoneName: string): string => {
  const { width, height } = COURT_DIMENSIONS;
  const wingLineY = height - (2 / 5) * height;

  // Calculate 3-point line intersection points
  const leftWingEndX =
    width / 2 -
    Math.sqrt(
      COURT_DIMENSIONS.threePointRadius * COURT_DIMENSIONS.threePointRadius -
        (height - 24 - wingLineY) * (height - 24 - wingLineY)
    );
  const rightWingEndX =
    width / 2 +
    Math.sqrt(
      COURT_DIMENSIONS.threePointRadius * COURT_DIMENSIONS.threePointRadius -
        (height - 24 - wingLineY) * (height - 24 - wingLineY)
    );

  switch (zoneName) {
    case "Restricted Area":
      return createRestrictedAreaPath();
    case "In The Paint (Non-RA)":
      return createPaintPath();
    case "Mid-Range":
      return createMidRangePath();
    case "Above the Break 3":
      return createAboveBreakThreePath(leftWingEndX, rightWingEndX, wingLineY);
    case "Left Corner 3":
      return createLeftCornerThreePath(leftWingEndX, wingLineY);
    case "Right Corner 3":
      return createRightCornerThreePath(rightWingEndX, wingLineY);
    default:
      return "";
  }
};

const createRestrictedAreaPath = (): string => {
  const { width, height, restrictedAreaRadius } = COURT_DIMENSIONS;
  return `M ${width / 2 - restrictedAreaRadius} ${height - 4 * 14.4} 
          A ${restrictedAreaRadius} ${restrictedAreaRadius} 0 0 1 ${width / 2 + restrictedAreaRadius} ${height - 4 * 14.4} 
          Z`;
};

const createPaintPath = (): string => {
  const { width, height, paintWidth, freeThrowLineDistance, restrictedAreaRadius } = COURT_DIMENSIONS;
  const paintLeft = (width - paintWidth) / 2;
  const paintRight = (width + paintWidth) / 2;
  const paintTop = height - 24 - freeThrowLineDistance;
  const paintBottom = height - 24;
  const restrictedY = height - 4 * 14.4;
  const centerX = width / 2;

  return `M ${paintLeft} ${paintTop} 
          L ${paintRight} ${paintTop} 
          L ${paintRight} ${paintBottom} 
          L ${centerX + restrictedAreaRadius} ${paintBottom}
          L ${centerX + restrictedAreaRadius} ${restrictedY}
          A ${restrictedAreaRadius} ${restrictedAreaRadius} 0 0 0 ${centerX - restrictedAreaRadius} ${restrictedY}
          L ${centerX - restrictedAreaRadius} ${paintBottom}
          L ${paintLeft} ${paintBottom}
          L ${paintLeft} ${paintTop} Z`;
};

const createMidRangePath = (): string => {
  const { width, height, paintWidth, freeThrowLineDistance, threePointRadius, threePointStraightDistance } = COURT_DIMENSIONS;
  const arcStartAngle = Math.asin((width / 2 - threePointStraightDistance) / threePointRadius);
  const arcStartX = width / 2 - Math.sin(arcStartAngle) * threePointRadius;
  const arcStartY = height - 24 - Math.cos(arcStartAngle) * threePointRadius;
  const arcEndX = width / 2 + Math.sin(arcStartAngle) * threePointRadius;

  const paintLeft = (width - paintWidth) / 2;
  const paintRight = (width + paintWidth) / 2;
  const paintTop = height - 24 - freeThrowLineDistance;
  const paintBottom = height - 24;

  return `M ${threePointStraightDistance} ${paintBottom}
          L ${threePointStraightDistance} ${arcStartY}
          L ${arcStartX} ${arcStartY}
          A ${threePointRadius} ${threePointRadius} 0 0 1 ${arcEndX} ${arcStartY}
          L ${width - threePointStraightDistance} ${arcStartY}
          L ${width - threePointStraightDistance} ${paintBottom}
          L ${paintRight} ${paintBottom}
          L ${paintRight} ${paintTop}
          L ${paintLeft} ${paintTop}
          L ${paintLeft} ${paintBottom}
          L ${threePointStraightDistance} ${paintBottom} Z`;
};

const createAboveBreakThreePath = (leftWingEndX: number, rightWingEndX: number, wingLineY: number): string => {
  const { width, height, threePointRadius, threePointStraightDistance } = COURT_DIMENSIONS;
  const arcStartAngle = Math.asin((width / 2 - threePointStraightDistance) / threePointRadius);
  const arcStartX = width / 2 - Math.sin(arcStartAngle) * threePointRadius;
  const arcStartY = height - 24 - Math.cos(arcStartAngle) * threePointRadius;
  const arcEndX = width / 2 + Math.sin(arcStartAngle) * threePointRadius;

  return `M ${leftWingEndX} ${wingLineY}
          L 0 ${wingLineY}
          L 0 0
          L ${width} 0
          L ${width} ${wingLineY}
          L ${rightWingEndX} ${wingLineY}
          L ${arcEndX} ${arcStartY}
          A ${threePointRadius} ${threePointRadius} 0 0 0 ${arcStartX} ${arcStartY}
          L ${leftWingEndX} ${wingLineY} Z`;
};

const createLeftCornerThreePath = (leftWingEndX: number, wingLineY: number): string => {
  const { width, height, threePointRadius, threePointStraightDistance } = COURT_DIMENSIONS;
  const arcStartAngle = Math.asin((width / 2 - threePointStraightDistance) / threePointRadius);
  const arcStartX = width / 2 - Math.sin(arcStartAngle) * threePointRadius;
  const arcStartY = height - 24 - Math.cos(arcStartAngle) * threePointRadius;
  
  const wingIntersectionAngle = Math.acos((height - 24 - wingLineY) / threePointRadius);
  const wingArcX = width / 2 - Math.sin(wingIntersectionAngle) * threePointRadius;
  
  return `M 0 ${wingLineY}
          L ${leftWingEndX} ${wingLineY}
          L ${wingArcX} ${height - 24 - Math.cos(wingIntersectionAngle) * threePointRadius}
          A ${threePointRadius} ${threePointRadius} 0 0 0 ${arcStartX} ${arcStartY}
          L ${threePointStraightDistance} ${arcStartY}
          L ${threePointStraightDistance} ${height - 24}
          L 0 ${height - 24}
          L 0 ${wingLineY} Z`;
};

const createRightCornerThreePath = (rightWingEndX: number, wingLineY: number): string => {
  const { width, height, threePointRadius, threePointStraightDistance } = COURT_DIMENSIONS;
  const arcStartAngle = Math.asin((width / 2 - threePointStraightDistance) / threePointRadius);
  const arcEndX = width / 2 + Math.sin(arcStartAngle) * threePointRadius;
  const arcEndY = height - 24 - Math.cos(arcStartAngle) * threePointRadius;
  
  const wingIntersectionAngle = Math.acos((height - 24 - wingLineY) / threePointRadius);
  const wingArcX = width / 2 + Math.sin(wingIntersectionAngle) * threePointRadius;
  
  return `M ${rightWingEndX} ${wingLineY}
          L ${width} ${wingLineY}
          L ${width} ${height - 24}
          L ${width - threePointStraightDistance} ${height - 24}
          L ${width - threePointStraightDistance} ${arcEndY}
          L ${arcEndX} ${arcEndY}
          A ${threePointRadius} ${threePointRadius} 0 0 0 ${wingArcX} ${height - 24 - Math.cos(wingIntersectionAngle) * threePointRadius}
          L ${rightWingEndX} ${wingLineY} Z`;
}; 