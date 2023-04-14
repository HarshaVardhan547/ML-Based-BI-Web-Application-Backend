import React from "react";

function Modeling() {
  return (
    <div className="aspect-w-16 aspect-h-16">
      <iframe
        src="https://streamlit.metriverse.systems/Insights"
        onLoad={() => {
          window.scrollTo(0, 0);
        }}
      ></iframe>
    </div>
  );
}

export default Modeling;
