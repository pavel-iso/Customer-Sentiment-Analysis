Customer Sentiment Analysis - Floweraura
1. Overview of Data Collection and Cleaning

Source: Reviews scraped from FlowerAura’s product page (10 pages of reviews).

Data Collected: Reviewer names, city, occasion, posted date, ratings, and review text.

Cleaning Steps:

Extracted and standardized review dates (removed suffixes like st, nd, rd, th).

Converted ratings from text to numeric (1–5 stars).

Applied TextBlob to compute:

Polarity → Sentiment score (–1 = negative, +1 = positive).

Subjectivity → Degree of opinion vs fact.

Added derived features: Review Length (word count) and Score (positive/negative based on polarity).

2. Sentiment Analysis Results

Distribution of Sentiment:

Majority of reviews are positive, reflecting overall customer satisfaction.

A smaller portion are negative, but those reviews tend to be longer and more detailed.

Average Sentiment per Rating:

5★ ratings* → Strongly positive polarity (customers happy with freshness, presentation, and delivery).

3★ ratings* → Mixed polarity (neutral or balanced opinions).

1★–2★ ratings* → Strongly negative polarity (complaints about delivery, freshness).

Correlation:

Positive correlation between numeric rating and sentiment polarity (higher rating = more positive sentiment).

Review Length Insights:

Negative reviews are longer → unhappy customers provide detailed complaints.

Positive reviews are shorter → often quick praises like “Beautiful bouquet” or “Loved it”.

3. Insights Positive Highlights:

Customers appreciate:

Freshness of roses.

Beautiful presentation of the bouquet.

On-time delivery on special occasions.

Emotional satisfaction when gifting (Anniversaries, Birthdays, Valentine’s Day).

Common Issues:

Late delivery reported in some cities.

Freshness not consistent (wilted flowers in a few cases).

Size/quantity mismatch (bouquet smaller than expected).

Packaging concerns (roses not arranged properly).

4. Recommendations

Product Quality:
Ensure strict freshness checks before dispatch.

Improve packaging to maintain bouquet shape during transit.

Delivery Experience:
Strengthen last-mile logistics to ensure timely delivery, especially on occasions.

Introduce delivery time-slot guarantees (customers value punctuality for gifting).

Customer Transparency:
Set clear expectations about bouquet size/quantity to avoid disappointment.

Highlight freshness guarantee in product descriptions and marketing.

Marketing Opportunities:
Leverage positive reviews (freshness, happiness of recipients) in ads.

Promote occasion-specific campaigns (Valentine’s Day, Anniversaries).

Offer loyalty discounts for repeat customers who gift flowers frequently.
