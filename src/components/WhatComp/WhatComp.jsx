import React from "react";
import "./WhatComp.css";
import { Feature } from "../../components";
import { WhatfeatureData } from "../../data/FeaturesData";

const WhatComp = () => {
  return (
    <div className="signlang__whatsignlang section__margin" id="whatsignlang">
      <div className="signlang__whatsignlang-feature">
        <Feature
          title="What is Sign Language"
          text="What is Sign Language? 🖐️  
Sign Language is a visual language using hand gestures, facial expressions, and body movements* to communicate. Indian Sign Language (ISL) is widely used by the Deaf community in India, enabling *inclusive and barrier-free communication. It’s a complete language* with its own grammar, making interactions seamless in daily life"
        />
      </div>

      <div className="signlang__whatsignlang-container">
        {WhatfeatureData.map((data, i) => (
          <Feature title={data.title} text={data.text} key={i * 201} />
        ))}
      </div>
    </div>
  );
};

export default WhatComp;
