import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface CountryData {
  [key: string]: number;
}

interface GeoJSONFeature {
  type: 'Feature';
  properties: {
    name?: string;
    name_long?: string;
    admin?: string;
    [key: string]: any;
  };
  geometry: any;
}

interface GeoJSONData {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}

const BasketballWorldMap: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [worldData, setWorldData] = useState<GeoJSONData | null>(null);

  // Basketball popularity data
  const basketballData: CountryData = {
    "USA": 100,
    "United States": 100,
    "United States of America": 100,
    "Philippines": 71,
    "Australia": 42,
    "Canada": 41,
    "Iraq": 32,
    "Mongolia": 29,
    "Lebanon": 26,
    "Lithuania": 23,
    "Macedonia": 22,
    "North Macedonia": 22,
    "New Zealand": 20,
    "Puerto Rico": 20,
    "Afghanistan": 19,
    "Greece": 16,
    "Bosnia and Herzegovina": 16,
    "Jamaica": 16,
    "Singapore": 15,
    "Fiji": 15,
    "Germany": 15,
    "Ghana": 15,
    "Estonia": 14,
    "Ireland": 13,
    "Zimbabwe": 13,
    "Uganda": 13,
    "Cameroon": 13,
    "Dominican Republic": 13,
    "Slovenia": 12,
    "Botswana": 12,
    "Georgia": 12,
    "Croatia": 12,
    "Trinidad and Tobago": 12,
    "Ethiopia": 11,
    "Nepal": 11,
    "Uruguay": 11,
    "Latvia": 10,
    "Hong Kong": 10,
    "United Kingdom": 10,
    "Nigeria": 10,
    "Tanzania": 10,
    "Kenya": 10,
    "Senegal": 9,
    "Albania": 9,
    "United Arab Emirates": 8,
    "Costa Rica": 8,
    "Honduras": 8,
    "Malaysia": 8,
    "Cote d'Ivoire": 8,
    "Côte d'Ivoire": 8,
    "Myanmar": 7,
    "Sri Lanka": 7,
    "Austria": 7,
    "Nicaragua": 7,
    "Mauritius": 7,
    "Panama": 7,
    "Israel": 6,
    "Indonesia": 6,
    "Switzerland": 6,
    "South Korea": 6,
    "Republic of Korea": 6,
    "Denmark": 6,
    "Kuwait": 6,
    "El Salvador": 6,
    "Jordan": 6,
    "Norway": 6,
    "Bulgaria": 6,
    "Guatemala": 6,
    "Ecuador": 5,
    "Mexico": 5,
    "India": 5,
    "Chile": 5,
    "Czechia": 5,
    "Czech Republic": 5,
    "Netherlands": 5,
    "Thailand": 5,
    "Belgium": 4,
    "Morocco": 4,
    "South Africa": 4,
    "Tunisia": 4,
    "Moldova": 4,
    "Syria": 4,
    "Slovakia": 4,
    "Bangladesh": 4,
    "Hungary": 4,
    "Oman": 4,
    "Portugal": 4,
    "Azerbaijan": 4,
    "Iran": 4,
    "Paraguay": 3,
    "Pakistan": 3,
    "Venezuela": 3,
    "Bolivia": 3,
    "Colombia": 3,
    "Peru": 3,
    "Sweden": 3,
    "France": 3,
    "Spain": 3,
    "Poland": 3,
    "Finland": 3,
    "Taiwan": 3,
    "Algeria": 3,
    "Romania": 3,
    "Turkey": 3,
    "Türkiye": 3,
    "Saudi Arabia": 3,
    "Egypt": 2,
    "Palestinian Territory": 2,
    "Palestine": 2,
    "Argentina": 2,
    "Italy": 2,
    "Belarus": 2,
    "Russian Federation": 2,
    "Russia": 2,
    "Viet Nam": 2,
    "Vietnam": 2,
    "Ukraine": 1,
    "Japan": 1,
    "China": 1,
    "Brazil": 1
  };

  const getCountryPopularity = (countryName: string): number => {
    return basketballData[countryName] || 0;
  };

  // Create color scale with darker shading for low values
  const colorScale = (value: number) => {
    // Normalize the value to 0-1 range but with a minimum threshold for visibility
    const minThreshold = 0.3; // Even the lowest values will be at least 30% intensity
    const normalizedValue = (value / 100) * (1 - minThreshold) + minThreshold;
    return d3.interpolateBlues(normalizedValue);
  };

  // Function to determine if text should be white or black based on background color brightness
  const getTextColor = (backgroundColor: string) => {
    // Convert the color to RGB values to calculate brightness
    const rgb = d3.rgb(backgroundColor);
    // Calculate relative luminance
    const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    // Return white text for dark backgrounds, black text for light backgrounds
    return luminance > 0.5 ? '#000000' : '#ffffff';
  };

  // Load data only once
  useEffect(() => {
    const loadMapData = async () => {
      try {
        console.log('Starting to load map data...');

        // Load GeoJSON world data from local file
        const worldResponse = await fetch('/ShootYoShot/data/countries-110m.json');
        console.log('Fetch response status:', worldResponse.status);
        
        if (!worldResponse.ok) {
          throw new Error('Failed to load world map data');
        }
        
        const data: GeoJSONData = await worldResponse.json();
        console.log('Loaded GeoJSON data:', data);
        console.log('Number of countries:', data.features.length);

        setWorldData(data);
        setLoading(false);

      } catch (err) {
        console.error('Error loading map data:', err);
        setError(`Failed to load map: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setLoading(false);
      }
    };

    loadMapData();
  }, []);

  // Render map when data is available and SVG is mounted
  useEffect(() => {
    if (!worldData || !svgRef.current || loading) {
      return;
    }

    console.log('SVG ref is available, proceeding with map rendering...');

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 720;
    const height = 450;

    // Create projection
    const projection = d3.geoNaturalEarth1()
      .scale(150)
      .translate([width / 2, height / 2])
      .center([0, 20]);  // Center the map vertically by shifting it up

    const path = d3.geoPath().projection(projection);

    // Create tooltip
    const tooltip = d3.select('body')
      .append('div')
      .attr('class', 'map-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('z-index', '1000');

    // Draw countries
    console.log('Drawing countries...');
    svg.selectAll('.country')
      .data(worldData.features)
      .enter()
      .append('path')
      .attr('class', 'country')
      .attr('d', (d: any) => path(d))
      .attr('fill', (d: any) => {
        const countryName = d.properties.name || d.properties.name_long || d.properties.admin || 'Unknown';
        const popularity = getCountryPopularity(countryName);
        return popularity > 0 ? colorScale(popularity) : '#eee';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event: MouseEvent, d: any) {
        const countryName = d.properties.name || d.properties.name_long || d.properties.admin || 'Unknown';
        const popularity = getCountryPopularity(countryName);
        
        d3.select(this)
          .attr('stroke-width', 2)
          .attr('stroke', '#333');
        
        tooltip
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`<strong>${countryName}</strong><br/>Basketball Popularity: ${popularity > 0 ? popularity : 'No data'}`);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke-width', 0.5)
          .attr('stroke', '#fff');
        
        tooltip.style('opacity', 0);
      });

    console.log('Map drawing completed');

    // Cleanup function
    return () => {
      d3.select('.map-tooltip').remove();
    };
  }, [worldData, loading]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      d3.select('.map-tooltip').remove();
    };
  }, []);

  if (loading) {
    return (
      <div className="w-full bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-2xl font-semibold mb-4 text-center">Basketball Popularity Worldwide</h3>
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">Loading world map...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="bg-white rounded-lg shadow-lg p-4 md:p-6">
        <h3 className="text-2xl md:text-3xl font-semibold mb-4 text-center text-black">Basketball Popularity Worldwide</h3>
        
        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {/* World Map container and Top Countries side by side - responsive */}
        <div className="flex flex-col xl:flex-row justify-center items-center gap-4 xl:gap-8 mb-6 w-full">
          {/* Map - responsive sizing */}
          <div className="flex-1 xl:flex-[2] w-full flex justify-center">
            <svg
              ref={svgRef}
              width="100%"
              height="450"
              viewBox="0 0 720 450"
              className="border border-gray-200 rounded max-w-full h-auto"
              style={{ background: '#f8fafc', minWidth: '600px' }}
            />
          </div>

          {/* Top countries list - responsive layout */}
          <div className="flex-shrink-0 w-full xl:w-auto xl:flex-[1] xl:min-w-[288px] flex flex-col items-center">
            <h4 className="text-lg md:text-xl font-semibold mb-3 text-center text-black leading-tight">
              Top 10 Countries by<br />Basketball Popularity
            </h4>
            <div className="flex flex-col gap-2 text-sm md:text-base w-full max-w-xs xl:w-72">
              {Object.entries(basketballData)
                .filter(([country, _]) => !["United States", "United States of America", "Republic of Korea", "North Macedonia", "Côte d'Ivoire", "Czech Republic", "Türkiye", "Palestine", "Russian Federation", "Vietnam"].includes(country))
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([country, popularity], index) => (
                  <div 
                    key={country} 
                    className="flex items-center justify-between p-2 rounded" 
                    style={{
                      backgroundColor: colorScale(popularity),
                      color: getTextColor(colorScale(popularity))
                    }}
                  >
                    <span className="font-semibold">{index + 1}. {country}</span>
                    <span>{popularity}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Legend - responsive */}
        <div className="mb-6">
          <p className="text-base md:text-lg text-gray-600 mb-3 text-center font-semibold">
            Regional Popularity Score (Basketball search interest relative to other sports):
          </p>
          <div className="flex flex-wrap justify-center items-center gap-2">
            <span className="text-sm md:text-base text-black">Low (1)</span>
            <div className="flex">
              {[1, 20, 40, 60, 80, 100].map((value) => (
                <div
                  key={value}
                  className="w-6 h-4 md:w-8 md:h-4"
                  style={{
                    backgroundColor: colorScale(value)
                  }}
                  title={`Popularity: ${value}`}
                />
              ))}
            </div>
            <span className="text-sm md:text-base text-black">High (100)</span>
            <div className="ml-2 md:ml-4 flex items-center gap-1">
              <div className="w-4 h-4 bg-gray-300"></div>
              <span className="text-sm md:text-base text-black">No data</span>
            </div>
          </div>
        </div>

        {/* Data source - responsive */}
        <div className="mb-6 text-sm md:text-base text-gray-500 text-center">
          <p>* Regional Popularity based on Google search traffic for 'Basketball' relative to other sports by country.</p>
          <p>Data source: TopEndSports.com (Google Insights for Search, 2007-2011)</p>
        </div>
      </div>
    </div>
  );
};

export default BasketballWorldMap; 