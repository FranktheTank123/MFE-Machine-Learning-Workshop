# Machine Learning in Real World

## Machine Learning Demystify

###  **Machine learning = statistics + coding + domain knowledge**

> Machine learning = 80% of data cleansing and feature engineering + 10% of `model.ﬁt()` + 10% of `model.predict()`  -- Rumors

* **Partially correct**
    * Data always 💩💩💩
    * A lot of model iterations
    * When to stop: how good is good?

### **Question**: how much ML do we need?

> **Interviewer**: You said you used SVM on 100-dimensional input variables. Tell me more about how to come up with these 100 dimensions.
> **Candidate**: My boss gave me the data.
> **Interviewer**: Okay. Why using SVM then?
> **Candidate**: My boss told me to do so.

**At least, you need to:**

- Know the derivation of basic models
- Hand-implement most basic models 
- Know the difference between models
- Know when to use what

### Machine Learning Skill Sets
![-c](images/ml_skillset.png) 

## Sample ML-Related Roles in the Wild
### Quant Researcher

#### Summary

* good input signals (garbage in, garbage out) 
* good strategies (extract alpha/beta from signals) and simple model
* bad strategy and very complex models (at least 10+ layer of deep recurrent/convolutional neural networks) 
* many fine-tunings
* pattern detections, factor analysis, clustering, etc
* Alternative data analysis

**Case study**: How to extract useful information from 10-Q/10-K's?

#### [Example 1: Citadel](https://www.citadel.com/careers/details/quantitative-researcher-full-time/)

>Quantitative Researchers play a key role on the Quantitative Research (“QR”) team, which is responsible for developing and testing automated quant trading strategies using sophisticated statistical techniques.
>
> **Your Objectives**
> 
> - Conceptualize valuation strategies, develop and continuously improve upon mathematical models, and help translate algorithms into code
> - **Back test** and **implement** trading models and signals in a live trading environment
> - Use **unconventional data sources** to drive innovation
> - Conduct **research** and **statistical analysis** to build and refine monetization systems for trading signals
>
> **Your Skills & Talents**
> 
> - Advanced training in Mathematics, Statistics, Physics, Computer Science, or another highly quantitative field (Bachelor’s, Master’s, PhD degree)
> - Strong knowledge of probability and statistics (e.g.,machine learning, time-series analysis, pattern recognition, NLP)
> - Prior experience working in a data driven research environment
> - Experience with NoSQL databases (e.g.,MongoDB)
> - Experience with distributed computing using MapReduce
> - Experience with analytical packages (e.g., R,Matlab) 
> - Independent research experience
> - Ability to manage multiple tasks and thrive in a fast-paced team environment
> - Excellent analytical skills, with strong attention to detail
> - Strong written and verbal communication skills

#### [Example 2: Two Sigma](https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Quantitative-Research-Associate/292)

> At Two Sigma, we’re different from other investment firms. Founded by a statistician and a computer scientist, our approach is systematic and diversified. The global markets fuel our imagination with unlimited information to study, countless efficient ways to take action and meaningful opportunity to improve by iteration. We draw ideas and inspiration from the broader math, science, and investment communities.
> 
> When you work with us, you tackle tough problems alongside other scientists and engineers. People who will challenge your ideas. Who you can really learn from, and collaborate with. And you’ll be doing work that matters to a lot of people, too. Our investors include some of the world’s largest retirement funds, research institutions, educational endowments, healthcare systems and foundations. We admire what they do, and we’re proud to serve these organizations.
> 
> **WHAT YOU’LL DO**
> As a quantitative researcher, you will:
> 
> - Use a rigorous scientific method to develop sophisticated investment models and shape our insights into how the markets will behave
> - Apply quantitative techniques like **machine learning** to **a vast array of datasets**
> - **Create** and **test** complex investment ideas and partner with our engineers to test your theories
> 
> All the while, you’ll remain engaged in the academic community. As examples, you can:
> 
> - Join our reading circles to stay up to date on the latest research papers in your fields
> - Attend academic seminars to learn from thought leaders from top universities
> - Share insights from conferences focused on statistics, machine learning, and data science
> 
> **WHAT WE’RE LOOKING FOR**
>  You’ll do best in this role if you:
> 
> - Have a degree in a technical or quantitative disciplines, like statistics, mathematics, physics, electrical engineering, or computer science (all levels welcome, from bachelor’s to doctorate)
> - Demonstrate intermediate skills in at least one programming language  (like C, C++, Java, or Python)
> - Performed an in-depth research project, examining real-world data
> - Are an independent thinker who can creatively approach data analysis and communicate complex ideas clearly
> 
> You don’t need a background in finance. It’s nice to have, but more than half of Two Sigma’s employees come from outside the finance industry. If you’ve got the quantitative skills, we can teach you the financial aspects of the job.

####  [Example 3: Blackrock SAE](https://www.velvetjobs.com/job-posting/blackrock-sae-quantitative-researcher-119321)

> With more than 85 people globally, the Scientific Active Equity (SAE) team has a major presence in San Francisco, New York, London, Tokyo and Sydney. The SAE team is focused on stock-selection strategies using fundamental insights captured in a scientific and rigorous process. Our clients include corporate pension plans, public pension plans, central banks, sovereign wealth funds and other institutional investors.
> 
> BlackRock’s Scientifically-driven Active Equity group is seeking a Researcher with the talent, drive, and entrepreneurial spirit to help us develop new and global active equity products. The successful candidate is passionately interested in investing: from researching ideas and testing them on historical data, to seeing which specific stocks the portfolio buys and sells as a result of those ideas. This candidate approaches all new ideas with an open mind, and a reliance on the scientific research process.
> 
> **Responsibilities:**
> 
> - Identify new active equity **ideas**, especially ideas orthogonal to those we use in existing BlackRock products.
> - Rigorously research new ideas to **determine their investment value in principle**, and their commercial value to BlackRock.
> - Seek out new sources of data to support this effort.
> - Work with a small, entrepreneurial team looking to quickly develop new products.
> - Communicate with other active equity researchers, to help Active Equity broadly achieve best practices.
> 
> **Skills and Qualifications:**
> 
> - Demonstrated ability to conduct high quality empirical research
> - Excellent quantitative skills, as evidenced by formal training in econometrics or statistics, and extensive experience in utilizing those skills in a research environment.
> - A strong understanding of equity markets, including drivers of return, risk control and portfolio construction techniques.
> - Strong academic training, e.g. graduate degree in finance, economics, or hard science.
> - Effective communication skills, both written and verbal. The successful candidate must have the confidence and credibility to persuasively interact with colleagues, management, and clients.
> - Strong understanding of computer technology in financial research, with strong programming skills.
> - Strong understanding of data available in the equity segment of the investment management industry, ability to ferret out relevant data, and ability to construct databases that are useful for validating research results.

    
### Data Scientist

#### Summary

* Improve productivities & sells, business analysis, making a better world
* know database very well
* have many user/business data (in order to use data-driven models)
* know how to query the data (efficiently!)
* be able to use okay models & their ensembles
* be a good story-teller

**Case study**: How to forecast ETA?

#### [Example 1: Facebook](https://www.facebook.com/careers/jobs/a0I1H00000LCNWKUA5/)

> Facebook's mission is to give people the power to build community and bring the world closer together. Through our family of apps and services, we're building a different kind of company that connects billions of people around the world, gives them ways to share what matters most to them, and helps bring people closer together. Whether we're creating new products or helping a small business expand its reach, people at Facebook are builders at heart. Our global teams are constantly iterating, solving problems, and working together to empower people around the world to build community and connect in meaningful ways. Together, we can help people build stronger communities — we're just getting started.
>
> We're looking for Data Scientists to work on our core and business products (ex. Instagram, Messaging, Growth, Engagement, Ads) to help shape the future of what we build at Facebook. You will enjoy working with one of the strongest data sets in the world, cutting edge technology, and the ability to see your insights turned into real products on a regular basis. The perfect candidate will have a background in a quantitative or technical field, will have experience working with large data sets, and will have some experience in data-driven decision making. You are focused on results, a self-starter, and have demonstrated success in using analytics to drive the understanding, growth, and success of a product. This position is based full time in our Seattle, WA office
> 
> **Responsibilities**
> 
> - Apply your expertise in quantitative analysis, data mining, and the presentation of data to see beyond the numbers and understand how our users interact with both our consumer and business products
> - Partner with Product and Engineering teams to solve problems and identify trends and opportunities
> - Inform, influence, support, and execute our product decisions and product launches
> - The Data Scientist Analytics role has work across the following four areas:
> - **Product Operations**
>   - Forecasting and setting product team goals
>   - Designing and evaluating experiments
>   - Monitoring key product metrics, understanding root causes of changes in metrics
>   - Building and analyzing dashboards and reports
>   - Building key data sets to empower operational and exploratory analysis
>   - Evaluating and defining metrics
> - **Exploratory Analysis**
>   - Proposing what to build in the next roadmap
>   - Understanding ecosystems, user behaviors, and long-term trends
>   - Identifying new levers to help move key metrics
>   - Building models of user behaviors for analysis or to power production systems
> - **Product Leadership**
>   - Influencing product teams through presentation of data-based recommendations
>   - Communicating state of business, experiment results, etc. to product teams
>   - Spreading best practices to analytics and product teams
> - **Data Infrastructure**
>   - Working in Hadoop and Hive primarily, sometimes MySQL, Oracle, and Vertica
>   - Automating analyses and authoring pipelines via SQL and python based ETL framework
> 
> **Minimum Qualifications**
> 
> - 2+ years experience doing quantitative analysis 
> - BA/BS in Computer Science, Math, Physics, Engineering, Statistics or other technical field
> - Experience in SQL or other programming languages
> - Development experience in any scripting language (PHP, Python, Perl, etc.)
> - Experience communicating the results of analyses with product and leadership teams to influence the strategy of the product
> - Knowledge of statistics (e.g., hypothesis testing, regressions)
> - Experience manipulating data sets through statistical software (ex. R, SAS) or other methods

####  [Example 2: Airbnb](https://www.airbnb.com/careers/departments/position/38405)

> Airbnb has become a global platform that connects travelers and hosts from over 34,000 cities. As such, it has collected a diverse set of data, which our Data Science team mines for insights that will propel our community and product forward. However, information is only as valuable as it is understood by decision-makers. We are looking for talented analysts, at all levels of experience and seniority, that can work cross-functionally with Data Scientists and business partners to translate insights to action.
> 
> The ideal candidate has an eye for detail, great communication, and a keenness for problem solving. Examples of projects you would work on include, but are not limited to, creating business critical dashboards, defining key metrics, and investigating challenging questions around user behavior. Working alongside Data Scientists who dig deep into Airbnb's data, you will help them translate complex findings and results into a compelling narrative. In this role, you will have tremendous upwards exposure as the Data Science team’s mouthpiece to senior business partners. If you’re passionate about leveraging data to drive business and product decisions, we want to hear from you.
> 
> **Responsibilities**
> 
> - **Ownership**: conceptualize, create, and maintain monitoring dashboards for tracking key metrics
> - **Communication**: design and draft regular reports of progress towards company goals to product groups and senior stakeholders
> - **Creativity**: refine and improve definition of metrics as the company’s challenges and data evolve
> - **Engineering**: collaborate with data engineers to better log data and manage the timeliness and accessibility of data tables
> - **Investigation**: carry out ad hoc descriptive analysis according to product needs, whether it be potential product opportunities or for debugging
> - **Empowerment**: strategize on making analyses easily repeatable and generalizable by other team members in the future
> 
> **Experience**
> 
> - **Must have:**
>   - Professional experience in data analysis and visualization
>   - Confidence with analytical tools such as Excel, R, Python, Stata,or Matlab
>   - Proven ability to succeed in both collaborative and independent work environments
>   - Expertise designing and delivering presentations
>   - 4+ years of applicable industry experience. 
> - **Also valuable:**
>   - Familiarity with SQL or other querying language
>   - Experience with a programming language
>   - Ability to work with dashboarding software such as Tableau or Google Analytics
>   - Leadership or managerial experience

    
### Machine Learning Engineer

#### Summary

* build advanced
infrastructures/pipelines
* automate everything!
* make Data Scientists' life easier

**Case study**: how to build a platform so that others can 1-click run many models (usually for tuning hyper-parameters)?

#### [Example 1: Opendoor](https://jobs.lever.co/opendoor/47bbc9cf-2d6e-4b5a-8f53-0050a1eb8937)

> At Opendoor, we’re on a mission to make it simple to buy and sell homes. The traditional process is broken, with an average home taking over 90 days to sell and costing thousands of dollars. We make buying and selling a home stress-free and instant. We’ve built an exceptional team, have raised over \$300 million from top-notch investors and are growing fast, buying and selling more than \$100 million of homes per month.
> 
> We are seeking an exceptional Software Engineer to work on Machine Learning services that power our pricing optimization, forecasting, inventory management, recommendations and other innovative systems. With a background in both software development and machine learning, you will collaborate with research and product teams to build prototypes, invent new features and deliver high quality data products. Every engineer at Opendoor has an outsized impact, and you'll lead the development of projects that define the future of the company.
> 
> **Your responsibilities may include:**
> 
> - Designing and developing scalable platforms/processes for model training, feature extraction, continuous learning, simulation and A/B testing
> - Building performant and expressive interfaces for the data and models
> - Contributing to all phases of algorithm development including ideation, prototyping, design and production
> - Applying machine learning and data mining techniques to create solutions to business problems
> - Become a domain expert in real-estate
> 
> **We’re looking for teammates who have:**
> 
> -  Experience building and productionizing end-to-end
> - machine learning applications 
> - Ability to write high performance production quality code
> - Experience in Python, Go, Java, Scala or other equivalent languages 
> - Good understanding of common families of models, feature engineering, feature selection and other practical machine learning concepts
> - Broad knowledge of machine learning APIs, tools, and open source libraries
> 
> **Bonus points:**
> 
> - Experience in distributed data processing frameworks such as Dask, Spark or Hadoop
> - Experience working with geodata or time series data 


