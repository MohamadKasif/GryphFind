-- One-time setup in a Snowflake worksheet. The app creates the POSTS table itself;
-- this just makes the database and shows the queries you can demo to judges.
CREATE DATABASE IF NOT EXISTS GRYPHFIND;
USE DATABASE GRYPHFIND;
USE SCHEMA PUBLIC;

-- Same table the app creates on startup
CREATE TABLE IF NOT EXISTS POSTS (
  ID STRING PRIMARY KEY, KIND STRING, TITLE STRING, DESCRIPTION STRING,
  LOCATION STRING, CONTACT STRING, CATEGORY STRING, IMAGE_PATH STRING,
  CREATED_AT FLOAT, STATUS STRING,
  IMG_VEC VECTOR(FLOAT, 512), TXT_VEC VECTOR(FLOAT, 512)
);

-- Demo query 1: best found-item match for every open lost post, computed in-warehouse
SELECT l.TITLE AS LOST, f.TITLE AS FOUND,
       ROUND(VECTOR_COSINE_SIMILARITY(l.IMG_VEC, f.IMG_VEC), 3) AS PHOTO_SIM,
       ROUND(VECTOR_COSINE_SIMILARITY(l.TXT_VEC, f.IMG_VEC), 3) AS WORDS_TO_PHOTO_SIM
FROM POSTS l JOIN POSTS f ON l.KIND = 'lost' AND f.KIND = 'found'
WHERE l.STATUS = 'open' AND f.STATUS = 'open'
QUALIFY ROW_NUMBER() OVER (PARTITION BY l.ID
  ORDER BY GREATEST(COALESCE(PHOTO_SIM, -1), COALESCE(WORDS_TO_PHOTO_SIM, -1) + 0.5) DESC) = 1;

-- Demo query 2: lost-and-found hotspots on campus
SELECT LOCATION, CATEGORY, COUNT(*) AS N
FROM POSTS GROUP BY LOCATION, CATEGORY ORDER BY N DESC;

-- Demo query 3: Cortex (open-weight Llama) summarising the board for campus staff
SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
  'In 3 bullet points, summarise what campus staff should know from these lost & found posts: ' ||
  LISTAGG(KIND || ': ' || TITLE || ' at ' || LOCATION, '; ')) AS SUMMARY
FROM POSTS WHERE STATUS = 'open';
