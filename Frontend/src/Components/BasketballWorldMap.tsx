import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import Frame from "./Frame";
import worldData from "../Assets/Data/countries-110m.json";
import popularities from "../Assets/Data/popularities.json";

const BasketballWorldMap: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);

  const getCountryPopularity = (countryName: string): number => {
    return popularities[countryName as keyof typeof popularities] || 0;
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
    return luminance > 0.5 ? "#000000" : "#ffffff";
  };

  // Render map when data is available and SVG is mounted
  useEffect(() => {
    if (!worldData || !svgRef.current) {
      return;
    }

    console.log("SVG ref is available, proceeding with map rendering...");

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 720;
    const height = 450;

    // Create projection
    const projection = d3
      .geoNaturalEarth1()
      .scale(150)
      .translate([width / 2, height / 2])
      .center([0, 20]); // Center the map vertically by shifting it up

    const path = d3.geoPath().projection(projection);

    // Create tooltip
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "map-tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", "1000");

    // Draw countries
    console.log("Drawing countries...");
    svg
      .selectAll(".country")
      .data(worldData.features)
      .enter()
      .append("path")
      .attr("class", "country")
      .attr("d", (d) => path(d as d3.GeoPermissibleObjects))
      .attr("fill", (d) => {
        const countryName =
          d.properties.name || d.properties.name_long || d.properties.admin || "Unknown";
        const popularity = getCountryPopularity(countryName);
        return popularity > 0 ? colorScale(popularity) : "#eee";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.5)
      .style("cursor", "pointer")
      .on("mouseover", function (event, d) {
        const countryName =
          d.properties.name || d.properties.name_long || d.properties.admin || "Unknown";
        const popularity = getCountryPopularity(countryName);

        d3.select(this).attr("stroke-width", 2).attr("stroke", "#333");

        tooltip
          .style("opacity", 1)
          .style("left", event.pageX + 10 + "px")
          .style("top", event.pageY - 10 + "px")
          .html(
            `<strong>${countryName}</strong><br/>Basketball Popularity: ${
              popularity > 0 ? popularity : "No data"
            }`
          );
      })
      .on("mouseout", function () {
        d3.select(this).attr("stroke-width", 0.5).attr("stroke", "#fff");

        tooltip.style("opacity", 0);
      });

    console.log("Map drawing completed");

    // Cleanup function
    return () => {
      d3.select(".map-tooltip").remove();
    };
  }, [worldData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      d3.select(".map-tooltip").remove();
    };
  }, []);

  return (
    <Frame midClass={"w-full min-w-[325px]"}>
      <div className="bg-base-300 rounded-lg shadow-lg p-4 md:p-6">
        <h3 className="text-2xl md:text-3xl font-semibold mb-4 text-center ">
          Basketball Popularity Worldwide
        </h3>

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
              style={{ background: "#f8fafc", minWidth: "600px" }}
            />
          </div>

          {/* Top countries list - responsive layout */}
          <div className="flex-shrink-0 w-full xl:w-auto xl:flex-[1] xl:min-w-[288px] flex flex-col items-center">
            <h4 className="text-lg md:text-xl font-semibold mb-3 text-center leading-tight">
              Top 10 Countries by
              <br />
              Basketball Popularity
            </h4>
            <div className="flex flex-col gap-2 text-sm md:text-base w-full max-w-xs xl:w-72">
              {Object.entries(popularities)
                .filter(
                  ([country]) =>
                    ![
                      "United States",
                      "United States of America",
                      "Republic of Korea",
                      "North Macedonia",
                      "Côte d'Ivoire",
                      "Czech Republic",
                      "Türkiye",
                      "Palestine",
                      "Russian Federation",
                      "Vietnam",
                    ].includes(country)
                )
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([country, popularity], index) => (
                  <div
                    key={country}
                    className="flex items-center justify-between p-2 rounded"
                    style={{
                      backgroundColor: colorScale(popularity),
                      color: getTextColor(colorScale(popularity)),
                    }}
                  >
                    <span className="font-semibold">
                      {index + 1}. {country}
                    </span>
                    <span>{popularity}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Legend - responsive */}
        <div className="mb-6">
          <p className="text-base md:text-lg text-gray-200 mb-3 text-center font-semibold">
            Regional Popularity Score (Basketball search interest relative to other sports):
          </p>
          <div className="flex flex-wrap justify-center items-center gap-2">
            <span className="text-sm md:text-base ">Low (1)</span>
            <div className="flex">
              {[1, 20, 40, 60, 80, 100].map((value) => (
                <div
                  key={value}
                  className="w-6 h-4 md:w-8 md:h-4"
                  style={{
                    backgroundColor: colorScale(value),
                  }}
                  title={`Popularity: ${value}`}
                />
              ))}
            </div>
            <span className="text-sm md:text-base ">High (100)</span>
            <div className="ml-2 md:ml-4 flex items-center gap-1">
              <div className="w-4 h-4 bg-gray-300"></div>
              <span className="text-sm md:text-base ">No data</span>
            </div>
          </div>
        </div>

        {/* Data source - responsive */}
        <div className="mb-6 text-sm md:text-base text-gray-300 text-center">
          <p>
            * Regional Popularity based on Google search traffic for 'Basketball' relative to other
            sports by country.
          </p>
          <p>Data source: TopEndSports.com (Google Insights for Search, 2007-2011)</p>
        </div>
      </div>
    </Frame>
  );
};

export default BasketballWorldMap;
