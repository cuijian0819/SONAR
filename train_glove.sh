cd /N/u/cuijian/BigRed200/security_event_detection/SONAR/glove/build;
./vocab_count -min-count 2 -verbose 2 < ../../data/tokenized_tweets.txt > vocab.txt;
./cooccur -memory 4.0 -vocab-file vocab.txt -verbose 2 -window-size 15 < ../../data/tokenized_tweets.txt > cooccurrence.bin
./glove -input-file cooccurrence.bin -vocab-file vocab.txt -save-file glove_sec_tweets_50d -vector-size 50 -window-size 15 -threads 4 -iter 30 -binary 2 -model 2
cp /N/u/cuijian/BigRed200/security_event_detection/SONAR/glove/build/glove_sec_tweets_50d.txt /N/u/cuijian/BigRed200/security_event_detection/SONAR/data