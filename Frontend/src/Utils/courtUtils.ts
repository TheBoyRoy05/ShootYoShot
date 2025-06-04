import { COURT_DIMENSIONS } from './courtConstants';

// Transform NBA coordinates to our court visualization using known data ranges
export const transformCoordinates = (x: number, y: number) => {
  // Known coordinate ranges from the dataset:
  // loc_x: -250 to 250 (500 units wide)
  // loc_y: -50 to 490 (540 units tall)
  const { width, height } = COURT_DIMENSIONS;

  // Map loc_x (-250 to 250) to screen x (0 to width)
  const normalizedX = (x - -250) / (250 - -250);

  // Map loc_y (-50 to 490) to screen y (height to 0, flipped)
  const normalizedY = (y - -50) / (490 - -50);

  return {
    x: normalizedX * width,
    y: height - normalizedY * height * 677/600, // Flip Y axis so y=490 is at top
  };
};

// Helper function to get zone color based on attempt percentage
export const getZoneColor = (attemptPercentage: number): string => {
  if (attemptPercentage === 0) return "rgba(0, 150, 0, 0.1)"; // Very light green for 0%

  // Use a more sensitive opacity calculation for better granularity
  // Square the normalized percentage to make differences more pronounced
  const maxAttemptPercentage = 0.4; // 40% is very high for any single zone
  const normalizedPercentage =
    Math.min(attemptPercentage, maxAttemptPercentage) / maxAttemptPercentage;
  const squaredPercentage = normalizedPercentage * normalizedPercentage; // Square for more dramatic differences
  const opacity = 0.15 + squaredPercentage * 0.65; // Range from 0.15 to 0.8

  return `rgba(0, 150, 0, ${opacity})`;
};

// Player name matching logic
export const matchesPlayerName = (shotPlayerName: string, selectedPlayer: string): boolean => {
  const name = shotPlayerName?.toLowerCase() || "";
  const selectedLower = selectedPlayer.toLowerCase();

  // Handle different name matching strategies
  if (selectedPlayer === 'Steph Curry') {
    return name.includes('curry') || name.includes('stephen') || name === 'stephen curry';
  } else if (selectedPlayer === 'Shaquille O\'Neal') {
    return name.includes('shaquille') || name.includes('o\'neal') || name.includes('oneal');
  } else if (selectedPlayer === 'Shai Gilgeous-Alexander') {
    return name.includes('shai') || name.includes('gilgeous') || name.includes('alexander');
  } else if (selectedPlayer === 'Jeremy Lin') {
    return name.includes('jeremy') && name.includes('lin');
  } else if (selectedPlayer === 'DeAndre Jordan') {
    return name.includes('deandre') || (name.includes('de') && name.includes('andre')) || name.includes('jordan');
  } else if (selectedPlayer === 'Giannis Antetokounmpo') {
    return name.includes('giannis') || name.includes('antetokounmpo');
  } else if (selectedPlayer === 'LeBron James') {
    return name.includes('lebron') || name.includes('james');
  } else if (selectedPlayer === 'Rudy Gobert') {
    return name.includes('rudy') && name.includes('gobert');
  } else if (selectedPlayer === 'Kobe Bryant') {
    return name.includes('kobe') || name.includes('bryant');
  }

  // Fallback: check if selected player name parts are in the shot player name
  const selectedParts = selectedLower.split(" ");
  return selectedParts.every((part) => name.includes(part));
}; 