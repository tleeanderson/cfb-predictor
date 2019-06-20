
#checking functions, these are defensive functions that make sure the values read in are correct
def check_team_stats(**kwargs):
    team_stats = kwargs['team_stats']

    correct, incorrect = [], []
    for k, v in team_stats.iteritems():
        if len(v.keys()) == 2:
            correct.append(k)
        else:
            incorrect.append(k)
    
    print("Correct len: %d, incorrect len: %d, total: %d", len(correct), len(incorrect), 
          len(correct) + len(incorrect))
