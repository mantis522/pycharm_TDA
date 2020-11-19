from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return score
    # return 1 if score['pos'] > score['neg'] else 0

text = "I recently bought this album to complete my collection of Alice in Chains albums. I must say this is my favorite AIC album by far. I would classify the album as a hard rock/metal sound and should not be lumped together with the work of Nirvana, Soundgarden and Pearl Jam or even AIC's later work. The songs are performed with a lot of raw emotion and creativity and there is a large amount of diversity of songs: all out metal 'We Die Young' to the jazzy 'Sea of Sorrow' and the upbeat 'Sunshine' and 'I Know Something (Bout You), this contrasts to 'Dirt' which seems to be very dense and dark. I truely don't feel there is any filler songs on the album. Finally, the sound quality of early/mid-90s CDs are absolutely supurb as they haven't been over-compressed yet. Thus, if you can get the 1990 vintage CD, the audio quality is sublime.If you enjoy Alice In Chains or early 90s rock, I would highly recommend this album."

text2 = "I recently bought this album to complete my collection of Alice in Chains albums . I must say this is my favorite AIC album by far . I would classify the album as a rock / metal sound and should not be lumped together with the work of Nirvana , Soundgarden and Pearl Jam or even AIC 's work . The songs are performed with a lot of raw emotion and creativity and there is a large amount of diversity of songs : all out metal ' We Die Young ' to the jazzy ' Sea of Sorrow ' and the upbeat ' Sunshine ' and 'I Know Something ( Bout You ) , this contrasts to ' Dirt ' which seems to be very dense and dark . I truely do not feel there is any filler songs on the album . Finally , the sound quality of early / mid - 90s CDs are absolutely supurb as they have not been over - compressed yet . Thus , if you can get the 1990 vintage CD , the audio quality is sublime . If you enjoy Alice In Chains or early 90s rock , I would highly recommend this album ."

text3 = "I loved this album . It is one of those rare CDs that I can start listening to again right after the last song plays . However , I will say that I was disappointed that there was not actually any Pete Seeger here . Anywhere . For an album titled \" The Seeger Sessions , \" I would have expected Bruce and Pete to sit down , shoot the bull , play a little , and have a picture taken of them smiling together . I do not even think Bruce went to visit the guy in the nine years it took to make this album . there is an awful lot on this disc , but Bruce pays more homage to Dixieland jazz than dear old Pete. To hear Bruce tell it ( on the liner notes ) , he had never heard of Folk music , and was shocked to sit down with a bunch of musicians and be able to play traditional songs right off the bat . Imagine ! Bottom line , this is an excellent album , despite Bruce coming off as a couple bricks short of a load ."

text4 = "I loved this album. It is one of those rare CDs that I can start listening to again right after the last song plays. However, I will say that I was disappointed that there was not actually any Pete Seeger here. Anywhere. For an album titled The Seeger Sessions, I would have expected Bruce and Pete to sit down, shoot the bull, play a little, and have a picture taken of them smiling together. I do not even think Bruce went to visit the guy in the nine years it took to make this album. There is an awful lot of good music on this disc, but Bruce pays more homage to Dixieland jazz than dear old Pete.To hear Bruce tell it (on the liner notes), he had never heard of Folk music, and was shocked to sit down with a bunch of musicians and be able to play traditional songs right off the bat. Imagine!Bottom line, this is an excellent album, despite Bruce coming off as a couple bricks short of a load."

text5 = "This book is a fascinating, yet tragic tale, of a man who not only invented a microscope that was ahead of it's time, but 'cures' for cancer and most other diseases, using a machine that used different frequencies. This man, Royal Rife, had to constantly battle with the corrupt Medical Cartel that is only concerned about the almighty dollar. It makes one sick to think about how much good his work could have been for world, except it was destroyed and is still being suppressed. I am glad that Barry Lynes wrote this book though, and hope that it will continue to spark peoples desires to carry on Rife's work.On the side, I would like to mention there is now more and more ongoing tests with RIFE type machines. And that quite advanced RIFE type machines can be bought nowadays. They are around $5000.00 US. Make sure they have a plasma tube and not just the pads if you want a genuine and more effective RIFE type machine."

text6 = "This book is a fascinating , yet tragic tale , of a man who not only invented a microscope that was ahead of it is time , but ' cures ' for cancer and most other diseases , using a machine that used different frequencies . This man , Royal Rife , had to constantly battle with the corrupt Medical Cartel that is only concerned about the almighty dollar . It makes one sick to think about how much good his work could have been for world , except it was destroyed and is still being suppressed . I am glad , and hope . On the side , I would like to mention there is now more and more ongoing tests with RIFE type machines . And that quite advanced RIFE type machines can be bought nowadays . They are around $ 5000.00 US . Make sure they have a plasma tube and not just the pads if you want a genuine and more effective RIFE type machine ."

a = vader_polarity(text5)

b = vader_polarity(text6)

print(a)
print(b)