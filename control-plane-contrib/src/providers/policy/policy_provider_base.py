# control-plane-contrib/enrollment/enrollment_provider_base.py

from abc import ABC, abstractmethod

class EnrollmentProvider(ABC):
    """
    Abstract base class for Enrollment Providers.
    Implementations must define how to retrieve and manage tenant/user enrollment metadata.
    """

    @abstractmethod
    def get_enrollment_status(self, tenant_id: str) -> str:
        """
        Returns the enrollment status for a given tenant.
        E.g., "pending", "active", "suspended".
        """
        pass

    @abstractmethod
    def update_enrollment_status(self, tenant_id: str, status: str) -> None:
        """
        Updates the enrollment status for the given tenant.
        """
        pass
