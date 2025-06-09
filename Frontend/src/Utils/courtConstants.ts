// Store court dimensions for zone calculations (NBA dimensions scaled to 14.4px per foot)
export const COURT_DIMENSIONS = {
  width: 864, // 50 feet -> 864px (720 * 1.2)
  height: 720, // 47 feet -> 720px (600 * 1.2)
  paintWidth: 276, // 16 feet -> 276px (230 * 1.2)
  paintHeight: 329, // 19 feet -> 329px (274 * 1.2)
  freeThrowLineDistance: 259, // 15 feet -> 259px (216 * 1.2)
  threePointRadius: 410, // 23'9" -> 410px (342 * 1.2)
  threePointStraightDistance: 52, // 3 feet -> 52px (43 * 1.2)
  restrictedAreaRadius: 70, // 4 feet -> 70px (58 * 1.2)
  basketRadius: 13, // 0.75 feet -> 13px (11 * 1.2)
  backboardWidth: 103, // 6 feet -> 103px (86 * 1.2)
  basketX: 432, // Center of court width (360 * 1.2)
  basketY: 721, // At baseline (601 * 1.2)
};

export const SVG_DIMENSIONS = {
  width: 936,
  height: 780,
};

export const PLAYERS = [
  'Steph Curry',
  'Shaquille O\'Neal',
  'Shai Gilgeous-Alexander',
  'Jeremy Lin',
  'DeAndre Jordan',
  'Giannis Antetokounmpo',
  'LeBron James',
  'Rudy Gobert',
  'Kobe Bryant',
  'Anthony Edwards'
];

export const ZONE_ORDER = [
  "Above the Break 3",
  "Mid-Range",
  "In The Paint (Non-RA)",
  "Left Corner 3",
  "Right Corner 3",
  "Restricted Area",
];

export const ZONE_LABELS = [
  { zone: 'Above the Break 3', displayName: 'Above the Break 3', x: 432, y: 108, anchor: 'middle' },
  { zone: 'Mid-Range', displayName: 'Mid-Range', x: 432, y: 360, anchor: 'middle' },
  { zone: 'In The Paint (Non-RA)', displayName: 'The Paint', x: 432, y: 504, anchor: 'middle' },
  { zone: 'Restricted Area', displayName: 'Restricted Area', x: 432, y: 612, anchor: 'middle' },
  { zone: 'Left Corner 3', displayName: 'Left Corner 3', x: 12, y: 480, anchor: 'start' },
  { zone: 'Right Corner 3', displayName: 'Right Corner 3', x: 852, y: 480, anchor: 'end' }
]; 