from smellclm.static_metrics import compute

JAVA = """
public int process(int a, int b, String c, List<Map<String, Integer>> d, boolean e) {
    // for (int i = 0; i < 10; i++) { if (x) { } }
    String s = "if (x) { while (y) { } }";
    for (int i = 0; i < a; i++) {
        if (b > i) {
            while (e) {
                try {
                    e = false;
                } catch (Exception ex) {
                    return -1;
                }
            }
        } else if (c != null) {
            Runnable r = () -> { System.out.println(c); };
        }
    }
    return 0;
}
"""


def test_java_metrics():
    m = compute(JAVA, "java")
    assert m["parameter_count"] == 5          # generic commas are not parameter separators
    assert m["max_nesting"] == 4              # for > if > while > try; lambda body is not control flow
    assert m["effective_loc"] == 17           # comment line and blank lines excluded


def test_python_metrics_self_elif_nested_def():
    code = '''
def f(self, a, b=1, *args, c, **kw):
    # comment
    if a:
        pass
    elif b:
        for x in a:
            with open(x) as fh:
                pass
    else:
        def inner():
            if a:
                if b:
                    if c:
                        pass
    return 1
'''
    m = compute(code, "python")
    assert m["parameter_count"] == 5          # self excluded; *args and **kw counted
    assert m["max_nesting"] == 3              # elif is a sibling; nested def ignored
    assert m["parser"] == "ast"


def test_cpp_void_and_go_receiver():
    assert compute("int f(void) { return 1; }", "cpp")["parameter_count"] == 0
    assert compute("func (s *Svc) Run(a int, b string) error { return nil }", "go")["parameter_count"] == 2


def test_typescript_this_param_and_template_string():
    code = "function f(this: Foo, a: number, b: Array<string>) { const s = `if (x) { ${a} }`; if (a) { } }"
    m = compute(code, "typescript")
    assert m["parameter_count"] == 2
    assert m["max_nesting"] == 1


def test_extract_brace_and_python_functions():
    from smellclm.extract import functions
    java = """class A {
    // void fake(int x) { }
    public int add(int a, int b) throws X {
        if (a > b) { return helper(a); }
        return a + b;
    }
    private static void run() { for (int i = 0; i < 2; i++) { go(i); } }
}"""
    fs = functions(java, "java")
    assert [(f.name, f.start_line, f.end_line) for f in fs] == [("add", 3, 6), ("run", 7, 7)]
    cpp = "int Shop::order(int a, std::string b) const {\n  return a;\n}\n"
    assert [(f.name, f.end_line) for f in functions(cpp, "cpp")] == [("order", 3)]
    py = "class C:\n    @d\n    def m(self, x):\n        return x\n\ndef g():\n    pass\n"
    assert [(f.name, f.start_line, f.end_line) for f in functions(py, "python")] == [("m", 2, 4), ("g", 6, 7)]
