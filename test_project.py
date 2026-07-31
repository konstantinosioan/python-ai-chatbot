from project import parse_command

def test_valid_command_without_argument():
    assert parse_command("/quit") == ("/quit", None)
    
    
def test_valid_command_with_argument():
    assert parse_command("/quit this is my program") == ("/quit", "this is my program")
    
    
def test_normal_input():
    assert parse_command("hello there") is None
    
    
def test_empty_input():
    assert parse_command("") is None
    

def test_invalid_command():
    assert parse_command("/bogus") is None
    
    
def test_valid_command_with_whitespace():
    assert parse_command(" /quit ") == ("/quit", None)