%load('ML101_KS.csv')
%load('s01.mat')
Fs=128;
arousal=[];
valence=[];
dominance=[];
liking=[];
kisi=32;
video=40;



for kisiSayisi=1:kisi % kiþi
    
  dosya= strcat( num2str(kisiSayisi),'.mat');
  load(dosya);
  
 
  
  
  
  arousal= [arousal ; labels(:,2)];
  valence= [valence ; labels(:,1)];
  dominance= [dominance ; labels(:,3)];
  liking= [liking ; labels(:,4)];
  
  
  
for videoSayisi=1:video% video

tekVideo=squeeze(data(videoSayisi,:,:));
%tekKanal=tekVideo(1,:);
%tekKanal=tekKanal(1,3*128+1:end);
cokKanal=tekVideo(1:32,:);
cokKanal=cokKanal(1:32,3*128+1:end);



dosya2= strcat( num2str(videoSayisi+(kisiSayisi-1)*40),'.csv');
%tekKanalCVS=ML101_KS(1,:);
csvwrite(dosya2,cokKanal);
end
end

%xx=arousal;


%%%%%%%%%%%%
gecici=[]
for k=1:video % video
    
for i=1:kisi % katýlýmcý her 41 katý kayýt eildi
   gecici=[gecici; arousal(k+40*(i-1))] ;
    
end

ort=mean(gecici);

for i=1:kisi % katýlýmcý 
    
    if(ort>=5)
    flag=1;
    else
    flag=0;
    end
    
    arousal(k+40*(i-1))=flag; %k=1 1 41 81
    
end

gecici=[];
end
%%%%%%%%%%%%
gecici=[]
for k=1:video % video
    
for i=1:kisi % katýlýmcý her 41 katý kayýt eildi
   gecici=[gecici; valence(k+40*(i-1))] ;
    
end

ort=mean(gecici);

for i=1:kisi % katýlýmcý 
    
    if(ort>=5)
    flag=1;
    else
    flag=0;
    end
    
    valence(k+40*(i-1))=flag; %k=1 1 41 81
    
end

gecici=[];
end
%%%%%%%%%%%%
gecici=[]
for k=1:video % video
    
for i=1:kisi % katýlýmcý her 41 katý kayýt eildi
   gecici=[gecici; dominance(k+40*(i-1))] ;
    
end

ort=mean(gecici);

for i=1:kisi % katýlýmcý 
    
    if(ort>=5)
    flag=1;
    else
    flag=0;
    end
    
    dominance(k+40*(i-1))=flag; %k=1 1 41 81
    
end

gecici=[];
end
%%%%%%%%%%%%
gecici=[]
for k=1:video % video
    
for i=1:kisi % katýlýmcý her 41 katý kayýt eildi
   gecici=[gecici; liking(k+40*(i-1))] ;
    
end

ort=mean(gecici);

for i=1:kisi % katýlýmcý 
    
    if(ort>=5)
    flag=1;
    else
    flag=0;
    end
    
    liking(k+40*(i-1))=flag; %k=1 1 41 81
    
end

gecici=[];

end




csvwrite('arousal.csv',arousal);
csvwrite('valence.csv',valence);
csvwrite('liking.csv',liking);
csvwrite('dominance.csv',dominance);














%save

%load(dosya2)