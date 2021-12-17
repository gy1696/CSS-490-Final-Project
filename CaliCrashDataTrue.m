%% Processing California Data from 2018 - 2021
tCali = readtable("CaliCrashData 2018-2021.csv");

tCali = tCali(:, ["collision_date", "collision_time", "county_location", "alcohol_involved", "killed_victims", "injured_victims", "party_count", "state_highway_indicator"]);

alcohol = tCali.alcohol_involved;
alcohol(isnan(tCali.alcohol_involved)) = 0;
tCali.alcohol_involved = alcohol;

tCali = sortrows(tCali, ["collision_date", "collision_time"]);

%% For the purposes of this dataset, March 19th, 2020 is the cutoff date
% for Pre Covid, given that it is the day a statewide shelter-in-place order was issued.
% https://calmatters.org/health/coronavirus/2021/03/timeline-california-pandemic-year-key-points/
PreCovid = tCali(1:1045364, :);
PostCovid = tCali(1045365:end, :);

writetable(PreCovid, 'CaliCrashDataPreCovid.csv');
writetable(PostCovid, 'CaliCrashDataPostCovid.csv');


PreCovid = readtable("CaliCrashDataPreCovid.csv");
PostCovid = readtable("CaliCrashDataPostCovid.csv");

%% Population estimates from https://www.census.gov/quickfacts/CA
%% Population of California as of July 1, 2019 is: 39,512,223

PreCovid.killed_victims = PreCovid.killed_victims / 39512223;
PreCovid.injured_victims = PreCovid.injured_victims / 39512223;
PreCovid = convertvars(PreCovid, {'county_location'}, 'categorical');

PreCovid.county_location = grp2idx(PreCovid.county_location);

PostCovid.killed_victims = PostCovid.killed_victims / 39512223;
PostCovid.injured_victims = PostCovid.injured_victims / 39512223;
PostCovid = convertvars(PostCovid, {'county_location'}, 'categorical');

Loc = unique(PostCovid.county_location);
LocID = grp2idx(Loc);
Loc = array2table(Loc);
LocID = array2table(LocID);
Loc(:,2) = LocID(:, 1);

PostCovid.county_location = grp2idx(PostCovid.county_location);


[s1, f1] = size(PreCovid);



% labels = {'date', 'time', 'county', 'alcohol involved', 'killed victims', 'injured victims', 'party count', 'highway status'};
% for i = 1:f1
%     for j = 1:f1
%         if i < j && j ~= 1 && j ~= 2
%             x = table2array(PreCovid(:, i));
%             y = table2array(PreCovid(:, j));
%             figure;
%             scatter(x, y);
%             xlabel(labels(:, i));
%             ylabel(labels(:, j));
%         end
%     end
% end

[s2, f2] = size(PostCovid);
labels = {'date', 'time', 'county', 'alcohol involved', 'killed victims', 'injured victims', 'party count', 'highway status'};
for i = 1:f1
    for j = 1:f1
        if i < j && j ~= 1 && j ~= 2
            x = table2array(PostCovid(:, i));
            y = table2array(PostCovid(:, j));
            figure;
            scatter(x, y);
            xlabel(labels(:, i));
            ylabel(labels(:, j));
        end
    end
end

% figure
% [mu, std] = normfit(PreCovid.killed_victims);
% xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
% plot(xKilled, normpdf(xKilled, mu, std),'b');
% hold on
% [mu, std] = normfit(PostCovid.killed_victims);
% xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
% plot(xKilled, normpdf(xKilled, mu, std),'r');
% title('Normal Distribution of Deaths')
% hold off
% 
% figure
% [mu, std] = normfit(PreCovid.alcohol_involved);
% xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
% plot(xKilled, normpdf(xKilled, mu, std),'b');
% hold on
% [mu, std] = normfit(PostCovid.alcohol_involved);
% xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
% plot(xKilled, normpdf(xKilled, mu, std),'r');
% title('Normal Distribution of Alcohol Involvement')
% hold off
