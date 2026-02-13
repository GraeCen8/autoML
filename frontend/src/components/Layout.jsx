import { Outlet, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LogOut, Database, FlaskConical, LayoutDashboard } from 'lucide-react';

const Layout = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    if (!user) return null;

    return (
        <div className="min-h-screen bg-background text-foreground flex">
            {/* Sidebar */}
            <aside className="w-64 border-r border-border p-6 flex flex-col">
                <h1 className="text-2xl font-bold mb-8 text-primary tracking-tight">AutoML</h1>

                <nav className="space-y-2 flex-1">
                    <NavLink to="/" icon={<LayoutDashboard size={20} />} label="Dashboard" />
                    <NavLink to="/databases" icon={<Database size={20} />} label="Databases" />
                    <NavLink to="/experiments" icon={<FlaskConical size={20} />} label="Experiments" />
                </nav>

                <div className="pt-6 border-t border-border">
                    <div className="mb-4">
                        <p className="text-sm font-medium">{user.username}</p>
                        <p className="text-xs text-muted-foreground">{user.email}</p>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="flex items-center space-x-2 text-muted-foreground hover:text-destructive transition-colors text-sm w-full"
                    >
                        <LogOut size={16} />
                        <span>Logout</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 p-8 overflow-y-auto">
                <Outlet />
            </main>
        </div>
    );
};

const NavLink = ({ to, icon, label }) => {
    return (
        <Link
            to={to}
            className="flex items-center space-x-3 px-4 py-3 rounded-md hover:bg-muted/50 transition-colors text-muted-foreground hover:text-foreground active:text-primary"
        >
            {icon}
            <span>{label}</span>
        </Link>
    );
}

export default Layout;
