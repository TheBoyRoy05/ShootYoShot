// Store court dimensions for zone calculations (NBA dimensions scaled to 14.4px per foot)
export const COURT_DIMENSIONS = {
  width: 720, // 50 feet -> 720px
  height: 600, // 47 feet -> 677px (adjusted)
  paintWidth: 230, // 16 feet -> 230px
  paintHeight: 274, // 19 feet -> 274px
  freeThrowLineDistance: 216, // 15 feet -> 216px
  threePointRadius: 342, // 23'9" -> 342px
  threePointStraightDistance: 43, // 3 feet -> 43px
  restrictedAreaRadius: 58, // 4 feet -> 58px
  basketRadius: 11, // 0.75 feet -> 11px
  backboardWidth: 86, // 6 feet -> 86px
  basketX: 360, // Center of court width
  basketY: 601, // At baseline
};

export const SVG_DIMENSIONS = {
  width: 780,
  height: 650,
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
  'Kobe Bryant'
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
  { zone: 'Above the Break 3', displayName: 'Above the Break 3', x: 360, y: 90, anchor: 'middle' },
  { zone: 'Mid-Range', displayName: 'Mid-Range', x: 360, y: 300, anchor: 'middle' },
  { zone: 'In The Paint (Non-RA)', displayName: 'The Paint', x: 360, y: 420, anchor: 'middle' },
  { zone: 'Restricted Area', displayName: 'Restricted Area', x: 360, y: 510, anchor: 'middle' },
  { zone: 'Left Corner 3', displayName: 'Left Corner 3', x: 10, y: 400, anchor: 'start' },
  { zone: 'Right Corner 3', displayName: 'Right Corner 3', x: 710, y: 400, anchor: 'end' }
]; 