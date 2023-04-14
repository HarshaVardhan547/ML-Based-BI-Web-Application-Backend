import React from "react";

function Upload() {
  return (
    <div className="aspect-w-16 aspect-h-16">
      <iframe
        src="https://streamlit.metriverse.systems/upload"
        onLoad={() => {
          window.scrollTo(0, 0);
        }}
      ></iframe>
    </div>
  );
}
export default Upload;
