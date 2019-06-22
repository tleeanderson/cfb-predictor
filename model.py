MAX_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=max)
MIN_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=min)
UNDECIDED_FIELDS = {'Field Goal Att', 'Rush Att', 'Time Of Possession', 'Kickoff', '1st Down Pass', 
'Red Zone Field Goal', 'Kickoff Onside', 'Def 2XP Att', 'Off XP Kick Made', '1st Down Rush', 'Def 2XP Made', 
'Pass TD', 'Field Goal Made', 'Pass Comp', 'Rush TD', 'Misc Ret', 'Kickoff Touchback', 'Off 2XP Att',
'Fourth Down Att', 'Off XP Kick Att', 'Pass Att', 'Off 2XP Made', 'Kickoff Ret', 'Pass Conv'}

FIELD_WIN_SEMANTICS = {('Red Zone Att', 'Tackle For Loss Yard', 'Fourth Down Conv', 'QB Hurry', 'Safety',
'Int Ret', 'Int Ret TD', 'Punt Ret Yard', 'Kickoff Yard', 'Fum Ret Yard', 'Pass Yard', 'Fumble Forced', 
'Red Zone TD', 'Misc Ret TD', 'Tackle Assist', 'Kickoff Ret TD', 'Punt Ret', 'Fum Ret', 'Points', 
'Misc Ret Yard', 'Pass Int', 'Int Ret Yard', 'Kick/Punt Blocked', 'Punt Yard', 
'Tackle For Loss', 'Kickoff Ret Yard', 'Third Down Conv', 'Fum Ret TD', 'Tackle Solo', 'Sack', 'Rush Yard',
'Sack Yard', 'Punt Ret TD', 'Pass Broken Up'): MAX_COMPARE,
('Kickoff Out-Of-Bounds', 'Penalty', 'Fumble', 'Penalty Yard', 'Punt', '1st Down Penalty', 
'Third Down Att', 'Fumble Lost'): MIN_COMPARE}

def attr_compare(**kwargs):
    v1, v2, compare = kwargs['v1'], kwargs['v2'], kwargs['compare']
    if v1 == v2:
        return 'tie'
    else:
        return [v1 + v2].index(compare(v1, v2))

# def eval_func(**kwargs):
#     stat_map1, stat_map2 = kwargs['stat_map1'], kwargs['stat_map2']

#     for sm1_key in stat_map1.keys():
        
