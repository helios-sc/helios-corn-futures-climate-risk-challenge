# Helios Corn Futures Climate Challenge - Official Competition Rules

## ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

**[See Section 3.18 for defined terms](rules#18.-terms)**

*The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules.*

**You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.**

## 1. COMPETITION-SPECIFIC TERMS

### 1. COMPETITION TITLE
Helios Corn Futures Climate Challenge

### 2. COMPETITION SPONSOR
Helios AI Inc.

### 3. COMPETITION SPONSOR ADDRESS
[INSERT HELIOS ADDRESS]

### 4. COMPETITION WEBSITE
https://www.kaggle.com/competitions/helios-corn-futures-climate-challenge

### 5. TOTAL PRIZES AVAILABLE: $25,000
- **First Prize**: $15,000
- **Second Prize**: $7,000  
- **Third Prize**: $3,000
- **Recognition**: Top 10 finishers receive Helios AI swag and potential interview opportunities

### 6. WINNER LICENSE TYPE
Open Source

### 7. DATA ACCESS AND USE
Competition Use and Non-Commercial & Academic Research

## 2. COMPETITION-SPECIFIC RULES

In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

### 1. TEAM LIMITS
a. The maximum Team size is five (5).

b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.

### 2. SUBMISSION LIMITS
a. You may submit a maximum of five (5) Submissions per day.

b. You may select up to two (2) Final Submissions for judging.

### 3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition's Overview > Timeline page.

### 4. COMPETITION DATA
a. **Data Access and Use**: You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.

b. **Data Security**: 
   1. You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

c. **Proprietary Climate Data**: The competition dataset contains proprietary climate risk assessments from Helios AI's climate risk model. Participants acknowledge that this data represents valuable intellectual property and agree to use it solely for competition purposes and permitted academic research.

### 5. WINNER LICENSE
a. Under Section 2.8 (Winners Obligations) of the General Rules below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

   1. **Open Source**: You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license (see [www.opensource.org](http://www.opensource.org)) that in no event limits commercial use of such code or model containing or depending on such code.

   2. For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, you do not need to grant the license in the preceding Section for that software.

   3. In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).

b. You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor's specifications, as outlined in Section 2.8, Winner's Obligations. This may include:
   - Detailed methodology description enabling reproduction
   - Feature engineering approach and rationale
   - Data preprocessing steps
   - Code repository with complete instructions
   - Explanation of novel climate-futures correlation techniques discovered

### 6. EXTERNAL DATA AND TOOLS
a. You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below.

b. **Reasonableness Standard**: The use of external data and models is acceptable unless specifically prohibited by the Host. External resources must be "reasonably accessible to all" and of "minimal cost". The Host will assess whether external LLMs, datasets, or tools meet the Reasonableness Standard considering:
   1. Cost thresholds and accessibility for all participants
   2. Whether participants are excluded due to "excessive" costs
   3. Example: Small subscription charges (e.g., Gemini Advanced) are acceptable; proprietary datasets exceeding the competition prize value are not reasonable

c. **Automated Machine Learning Tools ("AMLT")**: Individual Participants and Teams may use automated machine learning tools (e.g., Google AutoML, H2O Driverless AI, etc.) to create a Submission, provided that the Participant or Team ensures that they have an appropriate license to the AMLT such that they are able to comply with the Competition Rules.

d. **Climate Data Restrictions**: Participants may NOT use:
   - Other proprietary climate risk models or datasets
   - Real-time weather data feeds requiring paid subscriptions
   - Satellite imagery requiring commercial licenses
   - Any external climate data that provides unfair advantage over the provided Helios dataset

### 7. EVALUATION METRIC
a. **Climate-Futures Correlation Score (CFCS)**: Submissions will be evaluated using a custom metric that measures the strength of correlations between participant-engineered climate features and futures market variables.

b. **Scoring Formula**: CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)
   - Focus on significant correlations (≥ |0.5|) to reward quality over quantity
   - Scores range from 0-100, where higher is better

c. **Submission Format Requirements**: 
   - **Climate Features**: ALL engineered climate features must use the prefix `climate_risk_`
   - **Required Columns**: `date_on` (date), `country_name` (string), and optionally `region_name`
   - **Prohibited**: Modifying or including `futures_*` columns (provided by evaluation system)
   - **File Format**: CSV with proper headers and no missing required columns

d. **Naming Convention Enforcement**: The evaluation system automatically detects features by column prefixes. Submissions with incorrectly named features will receive zero points for those features. This is strictly enforced to ensure fair evaluation across all participants.

e. **ANTI-GAMING PROVISIONS**: 
   1. **Prohibited Data Leakage**: Participants may NOT use futures market data as input to create climate features. This includes:
      - Copying `futures_*` columns and renaming them with `climate_risk_` prefixes
      - Using futures prices, returns, or any derivatives as basis for climate features
      - Creating features that are mathematically derived from futures data
      - Any form of circular correlation where climate features are based on the target variables
   
   2. **Legitimate Feature Engineering**: Climate features must be derived solely from:
      - Original climate risk data provided in the dataset
      - External climate/weather data (subject to external data rules)
      - Mathematical transformations of legitimate climate data
      - Temporal aggregations of climate variables
      - Production-weighted climate metrics using regional market share data
   
   3. **Validation and Enforcement**: 
      - All submissions scoring above 80 will be manually reviewed for data leakage
      - Submissions found to violate these provisions will be disqualified
      - Participants engaging in systematic gaming may be banned from the competition
      - The Competition Sponsor reserves the right to implement automated detection of renamed futures columns
   
   4. **Realistic Score Expectations**: Legitimate climate-market correlations typically produce CFCS scores in the 20-70 range. Scores approaching 80+ often indicate data leakage or gaming rather than genuine climate insights.

### 8. ELIGIBILITY
a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means Helios AI Inc., Kaggle Inc., and their respective parent companies, subsidiaries and affiliates.

b. **Industry Restrictions**: Employees of competing agricultural technology companies, commodity trading firms, or weather/climate modeling companies may participate but are subject to additional disclosure requirements if they advance to prize-winning positions.

### 9. WINNER'S OBLIGATIONS
a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

   1. **Code Delivery**: Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code must:
      - Be capable of generating the winning Submission
      - Include feature engineering pipeline
      - Contain clear documentation and reproduction instructions
      - Include training code, inference code, and computational environment description

   2. **License Grant**: Grant to the Competition Sponsor the open source license to the winning Submission stated above, and represent that you have the unrestricted right to grant that license.

   3. **Documentation**: Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including:
      - Eligibility certifications
      - Licenses, releases and other agreements required under the Rules
      - U.S. tax forms (IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident)

   4. **Methodology Disclosure**: Provide a detailed technical writeup explaining:
      - Novel feature engineering approaches discovered
      - Climate-market relationship insights
      - Reproducible methodology description
      - Business implications of findings

### 10. INTELLECTUAL PROPERTY
a. **Helios Data**: The competition dataset remains the intellectual property of Helios AI Inc. Winners grant Helios the right to reference their methodologies in future research and product development.

b. **Participant Innovations**: Novel feature engineering techniques and climate-market correlation methods developed by participants will be open-sourced under the winner license terms.

### 11. PUBLICATION AND RESEARCH
a. **Academic Use**: Participants may publish academic papers based on their competition work, provided they:
   - Acknowledge Helios AI as the data provider
   - Do not redistribute the raw competition dataset
   - Share any published research with Helios AI

b. **Commercial Applications**: Winners may be invited to collaborate with Helios AI on commercial applications of their discoveries.

### 12. GOVERNING LAW
a. All claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts.

---

## Competition Objective

**Turn weather wisdom into trading gold!** Use Helios AI's proprietary climate data to decode the weather signals behind corn futures and outsmart the markets. Discover novel ways to transform climate risk assessments into features that show stronger correlations with commodity price movements.

**Good luck, and may the correlations be with you!** 🌽📈

---

*Last Updated: [INSERT DATE]*
*Competition Version: 1.0*