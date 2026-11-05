/**
 * Class Code Translator & Details Provider for ChilliGuard
 * Converts raw dataset codes (e.g. KTBM_5, KTWS_15) into human-readable descriptions.
 */

export const MULTICLASS_MAP = {
  KT00: {
    label: "Pure Khammam Teja Chilli",
    shortLabel: "Pure Chilli (0% Adulteration)",
    adulterant: "None (Unadulterated Control)",
    concentration: "0% w/w",
    category: "Pure",
    chilliVariety: "Khammam Teja Chilli",
    description: "Authentic, high-grade Khammam Teja red chilli powder. Contains zero fillers, sawdust, or synthetic dye."
  },
  KTBM_5: {
    label: "Brick Powder Adulteration (5% w/w)",
    shortLabel: "5% Brick Powder Adulterated",
    adulterant: "Brick Powder / Mortar Dust",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder fraudulently mixed with 5% fine brick powder. Brick dust is added to increase bulk weight and mimic the deep red color."
  },
  KTBM_10: {
    label: "Brick Powder Adulteration (10% w/w)",
    shortLabel: "10% Brick Powder Adulterated",
    adulterant: "Brick Powder / Mortar Dust",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder mixed with 10% fine brick dust. Significantly elevates insoluble ash content and causes gastrointestinal irritation."
  },
  KTBM_15: {
    label: "Brick Powder Adulteration (15% w/w)",
    shortLabel: "15% Brick Powder Adulterated",
    adulterant: "Brick Powder / Mortar Dust",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder mixed with high concentration (15%) brick dust. Results in noticeable gritty texture and high mineral residue."
  },
  KTGM_5: {
    label: "Groundnut Shell Meal (5% w/w)",
    shortLabel: "5% Groundnut Shell Adulterated",
    adulterant: "Groundnut Shell / Husk Meal",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder diluted with 5% pulverized groundnut shell husk as a cheap organic volume expander."
  },
  KTGM_10: {
    label: "Groundnut Shell Meal (10% w/w)",
    shortLabel: "10% Groundnut Shell Adulterated",
    adulterant: "Groundnut Shell / Husk Meal",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder diluted with 10% groundnut shell meal. Reduces natural capsaicin spiciness and introduces potential nut allergen risks."
  },
  KTGM_15: {
    label: "Groundnut Shell Meal (15% w/w)",
    shortLabel: "15% Groundnut Shell Adulterated",
    adulterant: "Groundnut Shell / Husk Meal",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder diluted with 15% groundnut shell meal. Causes severe loss of natural red color, aroma, and heat."
  },
  KTRB_5: {
    label: "Rice Bran / Husk (5% w/w)",
    shortLabel: "5% Rice Bran Adulterated",
    adulterant: "Rice Bran / Rice Chaff",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 5% finely ground rice bran husk to artificially pad product quantity at low cost."
  },
  KTRB_10: {
    label: "Rice Bran / Husk (10% w/w)",
    shortLabel: "10% Rice Bran Adulterated",
    adulterant: "Rice Bran / Rice Chaff",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 10% rice bran husk. Reduces natural oil content and alters spice density."
  },
  KTRB_15: {
    label: "Rice Bran / Husk (15% w/w)",
    shortLabel: "15% Rice Bran Adulterated",
    adulterant: "Rice Bran / Rice Chaff",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 15% rice bran husk. Results in pale coloration and high crude fiber residue."
  },
  KTUT_5: {
    label: "Spent Turmeric Powder (5% w/w)",
    shortLabel: "5% Spent Turmeric Adulterated",
    adulterant: "Spent / Waste Turmeric Powder",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder blended with 5% extracted spent turmeric powder byproduct to bulk up market volume."
  },
  KTUT_10: {
    label: "Spent Turmeric Powder (10% w/w)",
    shortLabel: "10% Spent Turmeric Adulterated",
    adulterant: "Spent / Waste Turmeric Powder",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder blended with 10% spent turmeric byproduct. Causes a yellowish hue shift in the red chilli powder."
  },
  KTUT_15: {
    label: "Spent Turmeric Powder (15% w/w)",
    shortLabel: "15% Spent Turmeric Adulterated",
    adulterant: "Spent / Waste Turmeric Powder",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder blended with 15% spent turmeric byproduct. Severe color distortion and loss of natural chilli aroma."
  },
  KTWB_5: {
    label: "Wheat Bran / Straw (5% w/w)",
    shortLabel: "5% Wheat Bran Adulterated",
    adulterant: "Wheat Bran / Wheat Straw",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 5% ground wheat bran flakes as an organic volume expander."
  },
  KTWB_10: {
    label: "Wheat Bran / Straw (10% w/w)",
    shortLabel: "10% Wheat Bran Adulterated",
    adulterant: "Wheat Bran / Wheat Straw",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 10% ground wheat bran flakes. Introduces gluten and dilutes natural heat."
  },
  KTWB_15: {
    label: "Wheat Bran / Straw (15% w/w)",
    shortLabel: "15% Wheat Bran Adulterated",
    adulterant: "Wheat Bran / Wheat Straw",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder adulterated with 15% ground wheat bran flakes. Creates a dull, flaky powder texture."
  },
  KTWS_5: {
    label: "Wood Sawdust (5% w/w)",
    shortLabel: "5% Wood Sawdust Adulterated",
    adulterant: "Wood Sawdust / Wood Shavings",
    concentration: "5% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder mixed with 5% fine wood sawdust. Added to pad product weight; floats during water immersion test."
  },
  KTWS_10: {
    label: "Wood Sawdust (10% w/w)",
    shortLabel: "10% Wood Sawdust Adulterated",
    adulterant: "Wood Sawdust / Wood Shavings",
    concentration: "10% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder mixed with 10% fine wood sawdust. High cellulose and fiber content; significant health and purity concern."
  },
  KTWS_15: {
    label: "Wood Sawdust (15% w/w)",
    shortLabel: "15% Wood Sawdust Adulterated",
    adulterant: "Wood Sawdust / Wood Shavings",
    concentration: "15% by weight",
    category: "Adulterated",
    chilliVariety: "Khammam Teja Chilli",
    description: "Chilli powder mixed with 15% fine wood sawdust. Heavy bulk contamination causing severe quality degradation."
  }
};

export function getClassDetails(classCode) {
  if (!classCode) return null;
  
  if (MULTICLASS_MAP[classCode]) {
    return {
      ...MULTICLASS_MAP[classCode],
      rawCode: classCode
    };
  }
  
  if (classCode === "Pure") {
    return {
      label: "Pure Chilli Powder",
      shortLabel: "Pure Chilli",
      adulterant: "None",
      concentration: "0%",
      category: "Pure",
      chilliVariety: "Red Chilli Powder",
      description: "Authentic chilli powder without detected adulteration.",
      rawCode: "Pure"
    };
  }
  
  if (classCode === "Adulterated") {
    return {
      label: "Adulterated Chilli Sample",
      shortLabel: "Adulterated",
      adulterant: "Impurity / Foreign Material Detected",
      concentration: "Adulterated Level",
      category: "Adulterated",
      chilliVariety: "Red Chilli Powder",
      description: "Sample shows significant deviation from pure chilli powder optical characteristics.",
      rawCode: "Adulterated"
    };
  }

  // Fallback for unknown codes
  return {
    label: classCode,
    shortLabel: classCode,
    adulterant: "Unknown Adulterant",
    concentration: "N/A",
    category: "Detected",
    chilliVariety: "Chilli Sample",
    description: `Sample classified as ${classCode}.`,
    rawCode: classCode
  };
}
